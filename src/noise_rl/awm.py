"""Dependency-light AWM WebSocket adapter; one session per rollout episode.

The AWM server is an OpenEnv application, but the trainer deliberately does
not import OpenEnv or ``agent_world_model_env``. Those packages currently
require NumPy 2 through ``mcp-agent``, while Megatron requires NumPy 1.x. This
module implements the small, public ``/ws`` protocol used by AWM instead.
"""
import asyncio
import json
from threading import Lock, Thread
from urllib.parse import urlsplit, urlunsplit

from .envs import StepResult

SYSTEM_PROMPT = '''Complete the task using the provided tools.
Reply with exactly one JSON object: {"tool_name":"name","arguments":{...}}.
Preserve case and whitespace in argument values. Use only listed tools.
When finished call {"tool_name":"done","arguments":{}}.
Tools can fail: inspect state before repeating an operation with an uncertain outcome.'''

_LOOP = None
_LOCK = Lock()


def canonical_action(text):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"tool_name", "arguments"}:
        raise ValueError("Expected tool_name and arguments")
    if not isinstance(value["tool_name"], str) or not value["tool_name"]:
        raise ValueError("tool_name must be nonempty")
    if not isinstance(value["arguments"], dict):
        raise ValueError("arguments must be an object")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _loop():
    global _LOOP
    with _LOCK:
        if _LOOP is None:
            _LOOP = asyncio.new_event_loop()
            Thread(target=_LOOP.run_forever, daemon=True, name="awm-io").start()
        return _LOOP


def _websocket_url(base_url):
    """Turn an AWM HTTP(S) base URL into its documented persistent ``/ws`` URL."""
    parsed = urlsplit(base_url)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parsed.scheme)
    if scheme is None or not parsed.netloc:
        raise ValueError(f"AWM URL must be an absolute http(s) or ws(s) URL: {base_url!r}")
    return urlunsplit((scheme, parsed.netloc, f"{parsed.path.rstrip('/')}/ws", parsed.query, ""))


class AWMWebSocketClient:
    """Minimal implementation of OpenEnv's public stateful ``/ws`` protocol.

    It intentionally uses only ``websockets`` and JSON. Keeping this boundary
    small prevents the trainer process from inheriting the AWM server's
    NumPy-2-only dependency graph.
    """

    def __init__(self, base_url, timeout=180, connect_factory=None):
        self.url = _websocket_url(base_url)
        self.timeout = timeout
        self._connect_factory = connect_factory
        self._websocket = None

    async def _connect(self):
        if self._websocket is not None:
            return
        connect = self._connect_factory
        if connect is None:
            try:
                from websockets.asyncio.client import connect
            except ImportError as exc:
                raise RuntimeError(
                    "AWM trainer client requires the lightweight 'websockets' package; "
                    "install this project in the Megatron training environment"
                ) from exc
        self._websocket = await connect(
            self.url,
            open_timeout=self.timeout,
            close_timeout=min(10, self.timeout),
            max_size=32 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )

    async def _request(self, message):
        await self._connect()
        assert self._websocket is not None
        await self._websocket.send(json.dumps(message, ensure_ascii=False, allow_nan=False))
        raw = await asyncio.wait_for(self._websocket.recv(), timeout=self.timeout)
        try:
            response = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"AWM server returned invalid JSON: {raw!r}") from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"AWM server returned a non-object response: {response!r}")
        if response.get("type") == "error":
            details = response.get("data") or {}
            if not isinstance(details, dict):
                details = {"message": str(details)}
            raise RuntimeError(
                "AWM WebSocket error: "
                f"{details.get('code', 'UNKNOWN')}: {details.get('message', details)!s}"
            )
        if response.get("type") != "observation" or not isinstance(response.get("data"), dict):
            raise RuntimeError(f"Unexpected AWM WebSocket response: {response!r}")
        return response["data"]

    async def reset(self, **kwargs):
        return await self._request({"type": "reset", "data": kwargs})

    async def list_tools(self):
        return await self._request({"type": "step", "data": {"type": "list_tools"}})

    async def call_tool(self, tool_name, arguments):
        return await self._request(
            {
                "type": "step",
                "data": {"type": "call_tool", "tool_name": tool_name, "arguments": arguments},
            }
        )

    async def close(self):
        websocket, self._websocket = self._websocket, None
        if websocket is None:
            return
        try:
            await websocket.send(json.dumps({"type": "close"}))
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


def _observation(result):
    """Extract the protocol observation while keeping old object tests useful."""
    if isinstance(result, dict):
        value = result.get("observation", result)
    else:
        value = getattr(result, "observation", result)
    if not isinstance(value, dict) and not hasattr(value, "__dict__"):
        raise RuntimeError(f"Unexpected AWM observation: {value!r}")
    return value


def _field(observation, name, default=None):
    if isinstance(observation, dict):
        return observation.get(name, default)
    return getattr(observation, name, default)


class AWMEnvironment:
    def __init__(self, task, url, timeout=180, client_factory=AWMWebSocketClient):
        self.task, self.url, self.timeout = task, url, timeout
        self.client_factory = client_factory
        self.client = None
        self.tools = set()

    canonical_action = staticmethod(canonical_action)

    def _run(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, _loop())
        try:
            return future.result(timeout=self.timeout)
        except BaseException:
            future.cancel()
            raise

    @staticmethod
    def _check(result):
        obs = _observation(result)
        kind = _field(obs, "reward_type")
        if kind in {"server_error", "timeout", "no_verifier", "reset_warning", "judge_error"}:
            raise RuntimeError(f"AWM infrastructure/verifier error: {kind}: {_field(obs, 'error', '')}")
        return obs

    async def _reset(self):
        self.client = self.client_factory(base_url=self.url, timeout=self.timeout)
        result = await self.client.reset(scenario=self.task["scenario"], task_idx=self.task["task_idx"])
        obs = self._check(result)
        has_code_verifier = (_field(obs, "has_verifier") or {}).get("code")
        if _field(obs, "reward_type") != "reset_ok" or not has_code_verifier:
            raise RuntimeError(f"AWM requires a successful reset and code verifier: {_field(obs, 'error', '')}")
        tools_result = await self.client.list_tools()
        tools_observation = self._check(tools_result)
        if _field(tools_observation, "error"):
            raise RuntimeError(f"AWM tool discovery failed: {_field(tools_observation, 'error')}")
        descriptions = [
            tool
            for tool in _field(tools_observation, "tools", [])
            if tool.get("name") not in {"verify", "done", "__list_scenarios__"}
        ]
        self.tools = {tool["name"] for tool in descriptions}
        return StepResult(json.dumps({"task": _field(obs, "task"), "tools": descriptions}, ensure_ascii=False))

    def reset(self):
        try:
            return self._run(self._reset())
        except BaseException:
            self.close()
            raise

    def is_read_only(self, action):
        # Explicit manifest allowlist: never infer semantics from tool names.
        return json.loads(action)["tool_name"] in self.task.get("read_only_tools", []) or json.loads(action)["tool_name"] == "done"

    async def _step(self, action):
        value = json.loads(action)
        name, arguments = value["tool_name"], value["arguments"]
        if name == "done":
            result = await self.client.call_tool("verify", {"verifier_mode": "code"})
            obs = self._check(result)
            if _field(obs, "reward_type") not in {"complete", "incomplete"}:
                raise RuntimeError(f"Unexpected AWM verifier outcome: {_field(obs, 'reward_type')}")
            return StepResult("Episode finished.", success=_field(obs, "reward_type") == "complete", terminated=True)
        if name not in self.tools:
            return StepResult("FORMAT_ERROR: unknown or reserved tool")
        result = await self.client.call_tool(name, arguments)
        obs = self._check(result)
        return StepResult(
            json.dumps({"result": _field(obs, "tool_result"), "error": _field(obs, "error")}, ensure_ascii=False)
        )

    def step(self, action):
        return self._run(self._step(action))

    def close(self):
        if self.client is not None:
            client, self.client = self.client, None
            try:
                self._run(client.close())
            except BaseException:
                pass
