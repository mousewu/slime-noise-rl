"""Optional OpenEnv AWM adapter; one persistent client session per episode."""
import asyncio
import json
from threading import Lock, Thread

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


class AWMEnvironment:
    def __init__(self, task, url, timeout=180):
        self.task, self.url, self.timeout = task, url, timeout
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
        obs = result.observation
        kind = getattr(obs, "reward_type", None)
        if kind in {"server_error", "timeout", "no_verifier", "reset_warning", "judge_error"}:
            raise RuntimeError(f"AWM infrastructure/verifier error: {kind}: {getattr(obs, 'error', '')}")
        return obs

    async def _reset(self):
        from agent_world_model_env import AWMEnv

        self.client = AWMEnv(base_url=self.url)
        await self.client.__aenter__()
        result = await self.client.reset(scenario=self.task["scenario"], task_idx=self.task["task_idx"])
        obs = self._check(result)
        if getattr(obs, "reward_type", None) != "reset_ok" or not (getattr(obs, "has_verifier", None) or {}).get("code"):
            raise RuntimeError("AWM requires a successful reset and code verifier")
        tools = await self.client.list_tools()
        descriptions = [t.model_dump() for t in tools if t.name not in {"verify", "done", "__list_scenarios__"}]
        self.tools = {t["name"] for t in descriptions}
        return StepResult(json.dumps({"task": obs.task, "tools": descriptions}, ensure_ascii=False))

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
        from openenv.core.env_server.mcp_types import CallToolAction

        value = json.loads(action)
        name, arguments = value["tool_name"], value["arguments"]
        if name == "done":
            result = await self.client.step(CallToolAction(tool_name="verify", arguments={"verifier_mode": "code"}))
            obs = self._check(result)
            if obs.reward_type not in {"complete", "incomplete"}:
                raise RuntimeError(f"Unexpected AWM verifier outcome: {obs.reward_type}")
            return StepResult("Episode finished.", success=obs.reward_type == "complete", terminated=True)
        if name not in self.tools:
            return StepResult("FORMAT_ERROR: unknown or reserved tool")
        result = await self.client.step(CallToolAction(tool_name=name, arguments=arguments))
        obs = self._check(result)
        return StepResult(json.dumps({"result": obs.tool_result, "error": obs.error}, ensure_ascii=False))

    def step(self, action):
        return self._run(self._step(action))

    def close(self):
        if self.client is not None:
            client, self.client = self.client, None
            self._run(client.close())
