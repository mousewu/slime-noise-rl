import asyncio
import json

import pytest

from noise_rl.awm import AWMEnvironment, AWMWebSocketClient, canonical_action
from noise_rl.config import NoiseConfig, load_config
from noise_rl.envs import StepResult
from noise_rl.noise import NoisyEnvironment


def test_awm_arguments_survive_noise_and_drop_does_not_execute():
    class Backend:
        canonical_action = staticmethod(canonical_action)
        seen = []
        def reset(self):
            return StepResult("ready")
        def is_read_only(self, action):
            return False
        def step(self, action):
            self.seen.append(json.loads(action))
            return StepResult("ok")

    backend = Backend()
    action = '{"tool_name":"WriteFile","arguments":{"text":"Hello  WORLD"}}'
    env = NoisyEnvironment(backend, NoiseConfig(0, 0), 1, 10)
    env.reset()
    env.step(action)
    assert backend.seen[0]["arguments"]["text"] == "Hello  WORLD"
    assert backend.seen[0]["tool_name"] == "WriteFile"
    env = NoisyEnvironment(backend, NoiseConfig(1, 0), 1, 10)
    env.reset()
    env.step(action)
    assert len(backend.seen) == 1


def test_awm_fails_closed_on_server_and_verifier_errors():
    for kind in ("server_error", "timeout", "no_verifier", "reset_warning"):
        with pytest.raises(RuntimeError):
            AWMEnvironment._check({"observation": {"reward_type": kind}})


def test_awm_configuration_is_separate():
    from pathlib import Path
    root = Path(__file__).parents[1]
    assert load_config(root / "configs/matched_loo_awm.yaml").awm_url
    assert load_config(root / "configs/matched_loo.yaml").awm_url is None


def test_awm_session_reset_tool_verify_and_close():
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        async def reset(self, **kwargs):
            calls.append(kwargs)
            return {
                "observation": {
                    "reward_type": "reset_ok",
                    "has_verifier": {"code": True},
                    "task": "Write text",
                }
            }
        async def list_tools(self):
            return {"observation": {"tools": [{"name": "write", "input_schema": {}}]}}
        async def call_tool(self, tool_name, arguments):
            calls.append((tool_name, arguments))
            return {
                "observation": {
                    "reward_type": "complete" if tool_name == "verify" else "tool_call_ok",
                    "tool_result": "ok",
                    "error": None,
                }
            }
        async def close(self):
            calls.append("closed")

    env = AWMEnvironment({"scenario": "test", "task_idx": 0}, "http://localhost:8899", client_factory=Client)
    try:
        assert "Write text" in env.reset().observation
        env.step(canonical_action('{"tool_name":"write","arguments":{"text":"A  B"}}'))
        result = env.step(canonical_action('{"tool_name":"done","arguments":{}}'))
        assert result.success and result.terminated
        assert calls[-1] == ("verify", {"verifier_mode": "code"})
        assert calls[-2] == ("write", {"text": "A  B"})
    finally:
        env.close()
    assert calls[-1] == "closed"


def test_awm_websocket_client_uses_public_openenv_protocol():
    messages = []

    class Socket:
        def __init__(self):
            self.responses = iter(
                [
                    '{"type":"observation","data":{"observation":{"reward_type":"reset_ok"},"done":false}}',
                    '{"type":"observation","data":{"observation":{"tools":[]},"done":false}}',
                    '{"type":"observation","data":{"observation":{"reward_type":"tool_call_ok"},"done":false}}',
                ]
            )
            self.closed = False

        async def send(self, payload):
            messages.append(json.loads(payload))

        async def recv(self):
            return next(self.responses)

        async def close(self):
            self.closed = True

    socket = Socket()

    async def connect(url, **kwargs):
        assert url == "ws://localhost:8899/ws"
        assert kwargs["max_size"] >= 32 * 1024 * 1024
        return socket

    async def exercise():
        client = AWMWebSocketClient("http://localhost:8899", connect_factory=connect)
        assert (await client.reset(scenario="demo", task_idx=2))["observation"]["reward_type"] == "reset_ok"
        await client.list_tools()
        await client.call_tool("write", {"text": "hello"})
        await client.close()

    asyncio.run(exercise())
    assert messages == [
        {"type": "reset", "data": {"scenario": "demo", "task_idx": 2}},
        {"type": "step", "data": {"type": "list_tools"}},
        {"type": "step", "data": {"type": "call_tool", "tool_name": "write", "arguments": {"text": "hello"}}},
        {"type": "close"},
    ]
    assert socket.closed
