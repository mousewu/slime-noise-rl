import json
from types import SimpleNamespace

import pytest

from noise_rl.awm import AWMEnvironment, canonical_action
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
            AWMEnvironment._check(SimpleNamespace(observation=SimpleNamespace(reward_type=kind)))


def test_awm_configuration_is_separate():
    from pathlib import Path
    root = Path(__file__).parents[1]
    assert load_config(root / "configs/matched_loo_awm.yaml").awm_url
    assert load_config(root / "configs/matched_loo.yaml").awm_url is None


def test_awm_session_reset_tool_verify_and_close(monkeypatch):
    import sys
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        async def __aenter__(self):
            return self
        async def reset(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(observation=SimpleNamespace(reward_type="reset_ok", has_verifier={"code": True}, task="Write text"))
        async def list_tools(self):
            return [SimpleNamespace(name="write", model_dump=lambda: {"name": "write", "input_schema": {}})]
        async def step(self, action):
            calls.append(action)
            return SimpleNamespace(observation=SimpleNamespace(reward_type="complete" if action.tool_name == "verify" else "tool_call_ok", tool_result="ok", error=None))
        async def close(self):
            calls.append("closed")

    monkeypatch.setitem(sys.modules, "agent_world_model_env", SimpleNamespace(AWMEnv=Client))
    monkeypatch.setitem(sys.modules, "openenv.core.env_server.mcp_types", SimpleNamespace(CallToolAction=SimpleNamespace))
    env = AWMEnvironment({"scenario": "test", "task_idx": 0}, "http://localhost:8899")
    try:
        assert "Write text" in env.reset().observation
        env.step(canonical_action('{"tool_name":"write","arguments":{"text":"A  B"}}'))
        result = env.step(canonical_action('{"tool_name":"done","arguments":{}}'))
        assert result.success and result.terminated
        assert calls[-1].arguments == {"verifier_mode": "code"}
        assert calls[-2].arguments == {"text": "A  B"}
    finally:
        env.close()
    assert calls[-1] == "closed"
