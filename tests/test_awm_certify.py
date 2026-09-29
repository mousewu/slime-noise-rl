import json

from noise_rl.awm import canonical_action
from noise_rl.awm_certify import certify_awm_environment, replay_row_diagnostics
from noise_rl.data import write_records
from noise_rl.envs import StepResult


def _task():
    return {
        "id": "awm/schedule/0",
        "environment": "awm",
        "split": "train",
        "scenario": "schedule",
        "task_idx": 0,
        "read_only_tools": [],
    }


def _manifest_record():
    return {"prompt": "Complete the tool-use task.", "metadata": {"task": _task()}}


def _action(tool_name, arguments):
    return canonical_action(json.dumps({"tool_name": tool_name, "arguments": arguments}))


def _replay_row(*, success, attempt=0):
    steps = []
    if success:
        steps.append(
            {
                "turn": 0,
                "action": _action("create_schedule", {"name": "Team"}),
                "format_error": False,
                "observation": '{"result":"created","error":null}',
                "environment_info": {"awm": {}},
            }
        )
    steps.append(
        {
            "turn": len(steps),
            "action": _action("done", {}),
            "format_error": False,
            "observation": "Episode finished.",
            "environment_info": {
                "awm": {
                    "verifier_reward_type": "complete" if success else "others",
                    "verify_execution_status": "success",
                }
            },
        }
    )
    return {
        "schema": 6,
        "phase": "awm_model_replay",
        "environment": "awm",
        "task": {**_task()},
        "attempt": attempt,
        "status": "success" if success else "completed_non_success",
        "success": success,
        "termination": "success" if success else "environment_terminal",
        "generated_tokens": 10,
        "format_errors": 0,
        "config": {
            "noise": {
                "action_drop": 0.0,
                "observation_loss": 0.0,
                "burst_probability": 0.0,
            }
        },
        "fault_audit": [],
        "steps": steps,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class _Environment:
    def __init__(self, task, _url, timeout):
        assert task["scenario"] == "schedule"
        assert timeout > 0
        self.created = False

    def reset(self):
        self.created = False
        return StepResult(
            json.dumps(
                {
                    "task": "Create a Team schedule.",
                    "tools": [
                        {
                            "name": "create_schedule",
                            "input_schema": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            },
                        }
                    ],
                },
                sort_keys=True,
            )
        )

    def step(self, action):
        value = json.loads(action)
        if value["tool_name"] == "create_schedule":
            self.created = value["arguments"] == {"name": "Team"}
            return StepResult('{"result":"created","error":null}')
        assert value["tool_name"] == "done"
        return StepResult(
            "Episode finished.",
            success=self.created,
            terminated=True,
            info={
                "awm": {
                    "verifier_reward_type": "complete" if self.created else "others",
                    "verify_execution_status": "success",
                }
            },
        )

    def close(self):
        pass


def test_nested_terminal_failure_is_not_mistaken_for_a_valid_attempt():
    row = _replay_row(success=False)
    row["steps"][0]["environment_info"] = {
        "awm": {
            "tool_terminal_failure": True,
            "tool_terminal_failure_type": "server_error",
        }
    }
    row["steps"][0]["observation"] = "ENVIRONMENT_ERROR: server_error: HTTP 500"

    diagnostics = replay_row_diagnostics(row)

    assert not diagnostics["valid_attempt"]
    assert any("tool_terminal_failure" in value for value in diagnostics["environment_errors"])


def test_certifier_emits_separate_environment_and_rl_manifests(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    witness = tmp_path / "teacher.jsonl"
    policy = tmp_path / "student.jsonl"
    write_records(manifest, [_manifest_record()])
    _write_jsonl(witness, [_replay_row(success=True)])
    _write_jsonl(
        policy,
        [_replay_row(success=index < 4, attempt=index) for index in range(16)],
    )
    env_output = tmp_path / "env.jsonl"
    rl_output = tmp_path / "rl.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    report = tmp_path / "report.json"

    result = certify_awm_environment(
        manifest=manifest,
        witness_replay=witness,
        policy_replay=policy,
        url="http://awm.invalid",
        environment_output=env_output,
        rl_output=rl_output,
        quarantine_output=quarantine,
        evidence_output=evidence,
        report_output=report,
        workers=1,
        reset_repeats=2,
        witness_repeats=2,
        environment_factory=_Environment,
    )

    assert result["tasks_environment_certified"] == 1
    assert result["tasks_rl_certified"] == 1
    assert len(env_output.read_text().splitlines()) == 1
    assert len(rl_output.read_text().splitlines()) == 1
    assert quarantine.read_text() == ""
    decision = json.loads(evidence.read_text())
    assert decision["live"]["structure"]["pass"]
    assert decision["live"]["witness"]["pass"]
    assert decision["live"]["verifier"]["pass"]
    assert decision["policy"]["success_rate"] == 0.25
    assert report.is_file()

    original_evidence = evidence.read_bytes()
    for path in (env_output, rl_output, quarantine, report):
        path.unlink()
    resumed = certify_awm_environment(
        manifest=manifest,
        witness_replay=witness,
        policy_replay=policy,
        url="http://awm.invalid",
        environment_output=env_output,
        rl_output=rl_output,
        quarantine_output=quarantine,
        evidence_output=evidence,
        report_output=report,
        workers=1,
        reset_repeats=2,
        witness_repeats=2,
        resume=True,
        environment_factory=_Environment,
    )
    assert resumed["tasks_rl_certified"] == 1
    assert evidence.read_bytes() == original_evidence


def test_too_easy_policy_is_environment_certified_but_not_rl_certified(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    witness = tmp_path / "teacher.jsonl"
    policy = tmp_path / "student.jsonl"
    write_records(manifest, [_manifest_record()])
    _write_jsonl(witness, [_replay_row(success=True)])
    _write_jsonl(policy, [_replay_row(success=True, attempt=index) for index in range(16)])

    result = certify_awm_environment(
        manifest=manifest,
        witness_replay=witness,
        policy_replay=policy,
        url="http://awm.invalid",
        environment_output=tmp_path / "env.jsonl",
        rl_output=tmp_path / "rl.jsonl",
        quarantine_output=tmp_path / "quarantine.jsonl",
        evidence_output=tmp_path / "evidence.jsonl",
        report_output=tmp_path / "report.json",
        workers=1,
        reset_repeats=1,
        witness_repeats=1,
        environment_factory=_Environment,
    )

    assert result["tasks_environment_certified"] == 1
    assert result["tasks_rl_certified"] == 0
    assert result["quarantine_reasons"] == {"policy_too_easy": 1}
