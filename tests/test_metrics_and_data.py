import copy
import hashlib
import json

import pytest

from noise_rl.data import mini_records, read_records, validate_local_records, write_records
from noise_rl.metrics import balanced_variance_components, paired_comparison, summarize, trace_metrics


def test_manifest_roundtrip_and_no_overwrite(tmp_path):
    path = tmp_path / "tasks.jsonl"
    rows = mini_records(3, "train")
    write_records(path, rows)
    assert read_records(path) == rows
    with pytest.raises(FileExistsError):
        write_records(path, rows)


def test_manifest_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "duplicates.jsonl"
    row = mini_records(1, "train")[0]
    write_records(path, [row, row])
    with pytest.raises(ValueError, match="unique"):
        read_records(path)


def test_local_alfworld_manifest_assets_are_verified(tmp_path):
    game = tmp_path / "game.tw-pddl"
    game.write_text(json.dumps({"solvable": True}), encoding="utf-8")
    digest = hashlib.sha256(game.read_bytes()).hexdigest()
    rows = [
        {
            "prompt": "Complete the household task.",
            "metadata": {
                "task": {
                    "id": "alfworld/train/example",
                    "environment": "alfworld",
                    "split": "train",
                    "gamefile": str(game),
                    "game_sha256": digest,
                }
            },
        }
    ]
    validate_local_records(rows)
    game.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_local_records(rows)


def test_variance_components_distinguish_fixed_environment_effects():
    values = balanced_variance_components([[0, 0, 0], [1, 1, 1]])
    assert values["within_environment_variance"] == 0
    assert values["between_environment_component_unclipped"] == 0.5
    values = balanced_variance_components([[0, 1], [0, 1]])
    assert values["between_environment_component_unclipped"] < 0


def episode(task_id, rank, success):
    from noise_rl.config import ExperimentConfig
    from noise_rl.sampling import plan_sample

    cfg = ExperimentConfig()
    return {
        "plan": plan_sample(cfg, task_id, 0, rank, evaluation=True).to_dict(),
        "config": cfg.to_dict(),
        "success": success,
        "termination": "success" if success else "turn_budget",
        "generated_tokens": 100,
        "inference_input_tokens": 300,
        "tool_calls": 5,
    }


def test_task_macro_metric_and_paired_ci():
    left = [episode(t, r, False) for t in ["a", "b"] for r in range(2)]
    right = copy.deepcopy(left)
    for row in right:
        row["success"] = True
    assert summarize(right)["success_rate_task_macro"] == 1
    delta = paired_comparison(left, right)
    assert delta["success_delta_right_minus_left"] == 1
    assert delta["task_paired_bootstrap_95ci"] == [1, 1]


def test_trace_metrics_covers_rollout_costs_and_fault_rates():
    records = [
        {
            "plan": {"task_id": "a"}, "success": True, "generated_tokens": 10,
            "inference_input_tokens": 20, "context_tokens": 30, "tool_calls": 2,
            "turns": 2, "format_errors": 0, "elapsed_seconds": 1.5,
            "termination": "success", "fault_audit": [{"dropped": True, "observation_lost": False}],
        },
        {
            "plan": {"task_id": "b"}, "success": False, "generated_tokens": 6,
            "inference_input_tokens": 9, "context_tokens": 15, "tool_calls": 1,
            "turns": 1, "format_errors": 1, "elapsed_seconds": 2.5,
            "termination": "tool_budget", "fault_audit": [{"dropped": False, "observation_lost": True}],
        },
    ]
    metrics = trace_metrics(records, prefix="rollout", step=7)
    assert metrics["rollout/step"] == 7
    assert metrics["rollout/success_rate"] == 0.5
    assert metrics["rollout/generated_tokens/total"] == 16
    assert metrics["rollout/faults/action_drop_rate"] == 0.5
    assert metrics["rollout/termination/success_rate"] == 0.5


def test_trace_metrics_include_awm_submission_and_schema_grounding_signals():
    records = [
        {
            "environment": "awm",
            "plan": {"task_id": "awm/a"},
            "success": False,
            "generated_tokens": 3,
            "inference_input_tokens": 5,
            "context_tokens": 8,
            "tool_calls": 1,
            "turns": 1,
            "format_errors": 0,
            "elapsed_seconds": 1,
            "termination": "environment_terminal",
            "fault_audit": [],
            "steps": [
                {
                    "action": '{"arguments":{"final_answer":"result"},"tool_name":"done"}',
                    "observation": "Episode finished.",
                    "environment_info": {"awm": {"verifier_reward_type": "others"}},
                }
            ],
        },
        {
            "environment": "awm",
            "plan": {"task_id": "awm/b"},
            "success": False,
            "generated_tokens": 4,
            "inference_input_tokens": 6,
            "context_tokens": 10,
            "tool_calls": 2,
            "turns": 2,
            "format_errors": 0,
            "elapsed_seconds": 2,
            "termination": "environment_terminal",
            "fault_audit": [],
            "steps": [
                {
                    "action": '{"arguments":{"value":null},"tool_name":"create"}',
                    "observation": "Input validation error: value must be an integer",
                    "environment_info": {},
                },
                {
                    "action": '{"arguments":{},"tool_name":"done"}',
                    "observation": "Episode finished.",
                    "environment_info": {"awm": {"verifier_reward_type": "complete"}},
                },
            ],
        },
    ]
    metrics = trace_metrics(records, prefix="rollout", step=8)
    assert metrics["rollout/awm/episodes"] == 2
    assert metrics["rollout/awm/done_first_rate"] == 0.5
    assert metrics["rollout/awm/final_answer_submitted_rate"] == 0.5
    assert metrics["rollout/awm/tool_input_validation_errors/total"] == 1
    assert metrics["rollout/awm/verifier/others_rate"] == 0.5
    assert metrics["rollout/awm/verifier/complete_rate"] == 0.5


@pytest.mark.parametrize("damage", ["seed", "conditions", "missing", "duplicate", "training"])
def test_invalid_paired_evaluation_rejected(damage):
    a = [episode("a", 0, False)]
    b = copy.deepcopy(a)
    if damage == "seed":
        b[0]["plan"]["environment_seed"] += 1
    elif damage == "conditions":
        b[0]["config"]["noise"]["action_drop"] = 0
    elif damage == "missing":
        b.clear()
    elif damage == "duplicate":
        b.append(b[0])
    else:
        b[0]["plan"]["evaluation"] = False
    with pytest.raises(ValueError):
        paired_comparison(a, b)
