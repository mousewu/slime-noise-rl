import json

from noise_rl.awm_eval_protocol import build_awm_eval_protocol
from noise_rl.awm_eval_summary import summarize_checkpoint_evaluations
from noise_rl.data import read_records, write_records


def _record(scenario, index):
    return {
        "prompt": "Complete the tool-use task.",
        "metadata": {"task": {
            "id": f"awm/{scenario}/{index}", "environment": "awm", "split": "train",
            "scenario": scenario, "task_idx": index, "read_only_tools": [],
        }},
    }


def test_build_protocol_and_summarize_disjoint_slices(tmp_path):
    environment = tmp_path / "environment.jsonl"
    rl = tmp_path / "rl.jsonl"
    write_records(environment, [_record("seen", 0), _record("seen", 1), _record("held", 0)])
    write_records(rl, [_record("seen", 0)])
    protocol_dir = tmp_path / "protocol"

    result = build_awm_eval_protocol(environment, rl, protocol_dir)

    assert result["counts"]["environment_non_rl"] == 2
    assert result["counts"]["scenario_heldout"] == 1
    assert len(read_records(protocol_dir / "replay_union.jsonl")) == 3

    replay = tmp_path / "replay.jsonl"
    rows = []
    for task_id, success in (("awm/seen/0", True), ("awm/seen/1", False), ("awm/held/0", True)):
        rows.append({
            "task": {"id": task_id}, "attempt": 0, "success": success,
            "status": "success" if success else "completed_non_success",
        })
    replay.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = summarize_checkpoint_evaluations(
        protocol_dir / "protocol.json", [("step1000", replay)],
        tmp_path / "summary.json", tmp_path / "summary.tsv", seed=7,
    )
    sets = report["checkpoints"]["step1000"]["sets"]
    assert sets["environment_non_rl"]["success_rate_task_macro"] == 0.5
    assert sets["scenario_heldout"]["success_rate_task_macro"] == 1.0
    assert sets["rl_resample"]["success_rate_task_macro"] == 1.0
