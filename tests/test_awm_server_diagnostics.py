import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from noise_rl import awm_server_diagnostics


def _fake_openenv_module(server_log: Path):
    class FakeProcess:
        def __init__(self):
            self._log_path = str(server_log)
            self._log_file = server_log.open("a", encoding="utf-8")

    class FakeAWMEnvironment:
        def __init__(self):
            self._process = FakeProcess()
            self._scenario = "threading_demo"
            self._task_idx = 3

        def _handle_call_tool(self, action, timeout_s=None):
            return SimpleNamespace(reward_type="server_error", error="Error calling create_thread: 500")

        def _handle_verify(self, action):
            return SimpleNamespace(
                reward_type="others",
                error="code verifier returned an unexpected status",
                verify_result={"reason": "broken verifier"},
            )

    module = ModuleType("agent_world_model_env.server.awm_environment")
    module.AWMEnvironment = FakeAWMEnvironment
    return module, FakeAWMEnvironment


def test_runtime_hook_preserves_tool_error_log_and_verifier_detail(tmp_path, monkeypatch, caplog):
    server_log = tmp_path / "session" / "server.log"
    server_log.parent.mkdir()
    server_log.write_text("server ready\nTraceback: sqlite table missing\nsecret=top-secret\n")
    module, environment_type = _fake_openenv_module(server_log)
    monkeypatch.setitem(sys.modules, "agent_world_model_env.server.awm_environment", module)
    monkeypatch.setenv("NOISE_RL_AWM_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setenv("HF_TOKEN", "top-secret")
    caplog.set_level(logging.ERROR, logger="noise_rl.awm_server_diagnostics")

    awm_server_diagnostics.install_awm_server_diagnostics()
    environment = environment_type()
    action = SimpleNamespace(tool_name="create_thread", arguments={"title": "test"})
    assert environment._handle_call_tool(action).reward_type == "server_error"
    assert environment._handle_verify(SimpleNamespace(tool_name="verify", arguments={})).reward_type == "others"
    environment._process._log_file.close()

    records = [
        json.loads(record.message.removeprefix("NOISE_RL_AWM_SUBPROCESS_DIAGNOSTIC "))
        for record in caplog.records
        if record.message.startswith("NOISE_RL_AWM_SUBPROCESS_DIAGNOSTIC ")
    ]
    assert [record["kind"] for record in records] == [
        "tool_server_error",
        "unexpected_verifier_outcome",
    ]
    assert records[0]["tool_name"] == "create_thread"
    assert records[0]["arguments"] == {"title": "test"}
    assert "Traceback: sqlite table missing" in records[0]["subprocess_log"]["tail"]
    assert "top-secret" not in records[0]["subprocess_log"]["tail"]
    assert records[1]["verify_result"] == {"reason": "broken verifier"}
    persisted = list((tmp_path / "diagnostics").glob("*.json"))
    assert len(persisted) == 2
    assert json.loads(persisted[0].read_text())["diagnostic_path"].endswith(".json")
