import json

from noise_rl import awm_smoke
from noise_rl.data import write_records
from noise_rl.envs import StepResult


def test_harness_smoke_forwards_final_answer_and_accepts_others(tmp_path, monkeypatch):
    manifest = tmp_path / "awm.jsonl"
    write_records(
        manifest,
        [
            {
                "prompt": "Complete task.",
                "metadata": {
                    "task": {
                        "id": "awm/demo/0",
                        "environment": "awm",
                        "split": "train",
                        "scenario": "demo",
                        "task_idx": 0,
                        "read_only_tools": [],
                    }
                },
            }
        ],
    )
    seen = []

    class Environment:
        def __init__(self, task, url, timeout):
            seen.append((task, url, timeout))

        def reset(self):
            return StepResult("ready")

        def step(self, action):
            payload = json.loads(action)
            assert payload["arguments"]["final_answer"] == "smoke answer"
            return StepResult(
                "finished",
                terminated=True,
                info={
                    "awm": {
                        "verifier_reward_type": "others",
                        "verify_execution_status": "success",
                        "final_answer_submitted": True,
                    }
                },
            )

        def close(self):
            pass

    monkeypatch.setattr(awm_smoke, "AWMEnvironment", Environment)
    output = tmp_path / "smoke.jsonl"
    summary = awm_smoke.run_harness_smoke(
        manifest, "http://127.0.0.1:8899", output, tasks=1, final_answer="smoke answer", timeout=12
    )
    assert summary["pass"] and summary["verifier_reward_types"] == {"others": 1}
    assert len(seen) == 1
    row = json.loads(output.read_text())
    assert row["terminated"] and row["final_answer_submitted"]
