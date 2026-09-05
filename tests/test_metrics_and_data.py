import copy
import hashlib
import json

import pytest

from noise_rl.data import mini_records, read_records, validate_local_records, write_records
from noise_rl.metrics import balanced_variance_components, paired_comparison, summarize


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
