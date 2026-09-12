import json
import logging
import sqlite3
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
            self._task = "Mark the queued thread as completed."
            self._tools_cache = [{"name": "update_thread", "inputSchema": {"type": "object"}}]
            self._trajectory = [
                {
                    "action": "call_tool",
                    "tool_name": "update_thread",
                    "arguments": {"id": "thread-7", "status": "completed"},
                    "success": True,
                    "result": {"id": "thread-7", "status": "completed"},
                    "error": None,
                },
                {"action": "verify", "reward_type": "others"},
            ]
            self._initial_db_path = str(server_log.parent / "initial.sqlite")
            self._db_path = str(server_log.parent / "final.sqlite")
            for path, status in ((self._initial_db_path, "queued"), (self._db_path, "completed")):
                with sqlite3.connect(path) as database:
                    database.execute("CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, status TEXT)")
                    database.execute("DELETE FROM threads")
                    database.execute("INSERT INTO threads VALUES (?, ?)", ("thread-7", status))

            class Loader:
                @staticmethod
                def get_verifier(scenario, task_idx, mode):
                    assert (scenario, task_idx, mode) == ("threading_demo", 3, "code")
                    return {
                        "verification": {
                            "code": "def verify_task_completion(initial_db_path, final_db_path, final_answer=None):\n    return {'result': 'others'}\n",
                            "success_criteria": "thread-7 must be completed",
                        }
                    }

            self._data_loader = Loader()

        def _handle_call_tool(self, action, timeout_s=None):
            return SimpleNamespace(reward_type="server_error", error="Error calling create_thread: 500")

        def _handle_verify(self, action):
            return SimpleNamespace(
                reward_type="others",
                error=None,
                verify_result={"execution_status": "success", "result": "others"},
            )

    module = ModuleType("agent_world_model_env.server.awm_environment")
    module.AWMEnvironment = FakeAWMEnvironment
    return module, FakeAWMEnvironment


def test_runtime_hook_preserves_tool_error_and_captures_verifier_evidence(tmp_path, monkeypatch, caplog):
    server_log = tmp_path / "session" / "server.log"
    server_log.parent.mkdir()
    server_log.write_text("server ready\nTraceback: sqlite table missing\nsecret=top-secret\n")
    module, environment_type = _fake_openenv_module(server_log)
    monkeypatch.setitem(sys.modules, "agent_world_model_env.server.awm_environment", module)
    monkeypatch.setenv("NOISE_RL_AWM_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setenv("NOISE_RL_AWM_EVIDENCE_MAX_BUNDLES", "1")
    monkeypatch.setenv("HF_TOKEN", "top-secret")
    caplog.set_level(logging.INFO, logger="noise_rl.awm_server_diagnostics")

    awm_server_diagnostics.install_awm_server_diagnostics()
    environment = environment_type()
    action = SimpleNamespace(tool_name="create_thread", arguments={"title": "test"})
    assert environment._handle_call_tool(action).reward_type == "server_error"
    assert environment._handle_verify(SimpleNamespace(tool_name="verify", arguments={})).reward_type == "others"
    environment._process._log_file.close()

    diagnostics = [
        json.loads(record.message.removeprefix("NOISE_RL_AWM_SUBPROCESS_DIAGNOSTIC "))
        for record in caplog.records
        if record.message.startswith("NOISE_RL_AWM_SUBPROCESS_DIAGNOSTIC ")
    ]
    assert [record["kind"] for record in diagnostics] == ["tool_server_error"]
    assert diagnostics[0]["tool_name"] == "create_thread"
    assert diagnostics[0]["arguments"] == {"title": "test"}
    assert "Traceback: sqlite table missing" in diagnostics[0]["subprocess_log"]["tail"]
    assert "top-secret" not in diagnostics[0]["subprocess_log"]["tail"]
    assert diagnostics[0]["diagnostic_path"].endswith(".json")

    markers = [
        json.loads(record.message.removeprefix("NOISE_RL_AWM_VERIFIER_EVIDENCE "))
        for record in caplog.records
        if record.message.startswith("NOISE_RL_AWM_VERIFIER_EVIDENCE ")
    ]
    assert len(markers) == 1
    assert markers[0]["reward_type"] == "others"
    assert markers[0]["evidence"]["status"] == "saved"
    evidence_path = Path(markers[0]["evidence"]["path"])
    evidence = json.loads(evidence_path.read_text())
    assert evidence["task"]["text"] == "Mark the queued thread as completed."
    assert "verify_task_completion" in evidence["verifier"]["source"]["text"]
    assert evidence["trajectory"]["captured_entries"] == 2
    assert evidence["database"]["diff"]["changed_tables"] == 1
    assert evidence["database"]["initial_backup"]["status"] == "saved"
    assert evidence["database"]["final_backup"]["status"] == "saved"
    assert (evidence_path.parent / "initial.sqlite").is_file()
    assert (evidence_path.parent / "final.sqlite").is_file()

    persisted = list((tmp_path / "diagnostics").glob("awm-diagnostic-*.json"))
    assert len(persisted) == 1
    assert json.loads(persisted[0].read_text())["diagnostic_path"].endswith(".json")
