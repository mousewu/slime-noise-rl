import hashlib
import json

import pytest

from noise_rl.awm import SYSTEM_PROMPT
from noise_rl.awm_sft_data import (
    AWM_SFT_DATASET,
    build_awm_sft_dataset,
    validate_awm_sft_records,
)
from noise_rl.sft_launch import validate_sft_input


def _action(tool_name, arguments):
    return json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
    )


def _success_row(*, initial_observation='{"task":"Create a schedule","tools":[]}', task_id="awm/schedule/0"):
    prompt_hash = hashlib.sha256(b"production-prompt").hexdigest()
    return {
        "schema": 6,
        "phase": "awm_model_replay",
        "environment": "awm",
        "status": "success",
        "success": True,
        "termination": "success",
        "config": {
            "noise": {
                "action_drop": 0.0,
                "observation_loss": 0.0,
                "burst_probability": 0.0,
                "burst_length": 3,
            }
        },
        "fault_audit": [],
        "task": {"id": task_id, "scenario": "schedule", "task_idx": 0, "split": "train"},
        "attempt": 0,
        "checkpoint": "iter_0000049",
        "generated_tokens": 11,
        "initial_observation": initial_observation,
        "initial_prompt_sha256": prompt_hash,
        "initial_prompt_tokens": 17,
        "steps": [
            {
                "turn": 0,
                "action": _action("create_schedule", {"name": "Team"}),
                "format_error": False,
                "observation": '{"result":"created"}',
                "environment_info": {"awm": {}},
            },
            {
                "turn": 1,
                "action": _action("done", {}),
                "format_error": False,
                "observation": "Episode finished.",
                "environment_info": {
                    "awm": {
                        "verifier_reward_type": "complete",
                        "verify_execution_status": "success",
                    }
                },
            },
        ],
    }


def _write_replay(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _accept_prompt(_row, _observation):
    return None


def test_builder_preserves_online_awm_turns_and_masks_only_actions(tmp_path):
    replay = tmp_path / "replay.jsonl"
    output = tmp_path / "awm-sft.jsonl"
    report = tmp_path / "awm-sft.report.json"
    _write_replay(replay, [_success_row()])

    value = build_awm_sft_dataset(replay, output, report, prompt_audit=_accept_prompt)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message["step_loss_mask"] for message in row["messages"]] == [0, 0, 1, 0, 1]
    assert row["messages"][0]["content"] == SYSTEM_PROMPT
    assert row["metadata"]["dataset"] == AWM_SFT_DATASET
    assert row["metadata"]["verifier"] == {"mode": "code", "reward_type": "complete"}
    assert value["records"] == 1
    assert value["initial_observation_source"] == {"stored": 1}
    assert validate_awm_sft_records(output)["expert_actions"] == 2
    assert validate_sft_input(output)["dataset"] == AWM_SFT_DATASET


def test_builder_rehydrates_legacy_rows_and_can_prompt_audit(tmp_path):
    replay = tmp_path / "legacy.jsonl"
    output = tmp_path / "legacy-sft.jsonl"
    report = tmp_path / "legacy-sft.report.json"
    row = _success_row(initial_observation=None)
    _write_replay(replay, [row])
    audited = []

    def audit(source, observation):
        audited.append((source["task"]["id"], observation))

    value = build_awm_sft_dataset(
        replay,
        output,
        report,
        initial_observation_resolver=lambda task: '{"task":"replayed","tools":[]}',
        prompt_audit=audit,
    )

    assert audited == [("awm/schedule/0", '{"task":"replayed","tools":[]}')]
    assert value["initial_observation_source"] == {"rehydrated": 1}


def test_builder_rejects_noisy_or_non_code_verifier_successes(tmp_path):
    replay = tmp_path / "bad.jsonl"
    output = tmp_path / "bad-sft.jsonl"
    report = tmp_path / "bad-sft.report.json"
    noisy = _success_row(task_id="awm/schedule/1")
    noisy["config"]["noise"]["action_drop"] = 0.15
    invalid_verifier = _success_row(task_id="awm/schedule/2")
    invalid_verifier["steps"][-1]["environment_info"] = {"awm": {"verifier_reward_type": "others"}}
    _write_replay(replay, [noisy, invalid_verifier])

    with pytest.raises(ValueError, match="No clean"):
        build_awm_sft_dataset(replay, output, report, prompt_audit=_accept_prompt)
    assert not output.exists()
    assert not report.exists()


def test_builder_requires_exact_online_prompt_audit(tmp_path):
    replay = tmp_path / "replay.jsonl"
    output = tmp_path / "awm-sft.jsonl"
    report = tmp_path / "awm-sft.report.json"
    _write_replay(replay, [_success_row()])

    with pytest.raises(ValueError, match="prompt_audit is required"):
        build_awm_sft_dataset(replay, output, report)


def test_builder_limits_duplicate_task_to_shortest_success(tmp_path):
    replay = tmp_path / "repeated.jsonl"
    output = tmp_path / "repeated-sft.jsonl"
    report = tmp_path / "repeated-sft.report.json"
    first = _success_row()
    second = _success_row()
    second["steps"].insert(
        1,
        {
            "turn": 1,
            "action": _action("list_items", {}),
            "format_error": False,
            "observation": '{"items":[]}',
            "environment_info": {"awm": {}},
        },
    )
    _write_replay(replay, [second, first])

    value = build_awm_sft_dataset(replay, output, report, prompt_audit=_accept_prompt)

    assert value["records"] == 1
    assert value["rejected"]["per_task_cap"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["metadata"]["action_count"] == 2
