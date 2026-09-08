import hashlib
import json

import pytest

from noise_rl.envs import StepResult
from noise_rl.sft_data import (
    action_json,
    build_messages,
    build_sft_dataset,
    validate_sft_records,
    walkthrough_from_game,
)


class ExpertFixtureEnvironment:
    def __init__(self, task):
        self.task = task
        self.actions = []
        self.closed = False

    def reset(self):
        return StepResult("Task: put apple 1 on table 1.")

    def step(self, action):
        self.actions.append(action)
        if action == "take apple 1 from shelf 1":
            return StepResult("You take apple 1.")
        if action == "put apple 1 in/on table 1":
            return StepResult("You put apple 1 on table 1.", success=True, terminated=True)
        return StepResult("Nothing happens.")

    def is_read_only(self, action):
        return action == "look"

    def close(self):
        self.closed = True


def task_record(tmp_path, *, walkthrough=None):
    game = tmp_path / "game.tw-pddl"
    raw = {"solvable": walkthrough is not None, "walkthrough": walkthrough}
    game.write_text(json.dumps(raw), encoding="utf-8")
    digest = hashlib.sha256(game.read_bytes()).hexdigest()
    task = {
        "id": "alfworld/train/pick_and_place/fixture",
        "environment": "alfworld",
        "split": "train",
        "gamefile": str(game),
        "game_sha256": digest,
        "task_type": "pick_and_place_simple",
    }
    return {"prompt": "Complete the household task.", "metadata": {"task": task}}


def test_stored_walkthrough_becomes_masked_multi_turn_messages(tmp_path):
    record = task_record(tmp_path, walkthrough=["take apple 1 from shelf 1", "put apple 1 in/on table 1"])
    walkthrough = walkthrough_from_game(record["metadata"]["task"])
    assert walkthrough == ["take apple 1 from shelf 1", "put apple 1 in/on table 1"]

    messages, actions = build_messages(record["metadata"]["task"], walkthrough, ExpertFixtureEnvironment)

    assert actions == 2
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user", "assistant"]
    assert [message["step_loss_mask"] for message in messages] == [0, 0, 1, 0, 1]
    assert messages[2]["content"] == action_json("take apple 1 from shelf 1")


def test_builder_writes_only_replay_verified_rows_and_audit_report(tmp_path):
    record = task_record(tmp_path, walkthrough=["take apple 1 from shelf 1", "put apple 1 in/on table 1"])
    output = tmp_path / "expert.jsonl"

    report = build_sft_dataset([record], output, environment_factory=ExpertFixtureEnvironment)

    assert report["verified_records"] == 1
    assert report["expert_actions"] == 2
    assert report["sources"] == {"game.tw-pddl.walkthrough": 1}
    facts = validate_sft_records(output)
    assert facts["records"] == 1 and facts["expert_actions"] == 2
    assert json.loads((tmp_path / "expert.jsonl.report.json").read_text(encoding="utf-8"))["dataset_sha256"] == facts[
        "sha256"
    ]
    with pytest.raises(FileExistsError):
        build_sft_dataset([record], output, environment_factory=ExpertFixtureEnvironment)


def test_builder_excludes_plan_that_does_not_reach_terminal_success(tmp_path):
    record = task_record(tmp_path, walkthrough=["take apple 1 from shelf 1"])

    with pytest.raises(ValueError, match="No verified expert trajectories"):
        build_sft_dataset([record], tmp_path / "invalid.jsonl", environment_factory=ExpertFixtureEnvironment)


def test_sft_validator_rejects_unmasked_or_noncanonical_assistant_action(tmp_path):
    row = {
        "messages": [
            {"role": "system", "content": "system", "step_loss_mask": 0},
            {"role": "user", "content": "observation", "step_loss_mask": 0},
            {"role": "assistant", "content": '{"action": "look"}', "step_loss_mask": 0},
        ],
        "metadata": {"schema": 1, "dataset": "alfworld_planner_sft", "task_id": "task", "split": "train"},
    }
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="step_loss_mask"):
        validate_sft_records(path)
