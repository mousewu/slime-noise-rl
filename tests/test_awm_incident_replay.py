import json

import pytest

from noise_rl.awm_incident_replay import load_incidents


def test_load_incidents_keeps_reviewed_task_requirement_and_deduplicates(tmp_path):
    source = tmp_path / "incidents.jsonl"
    row = {
        "kind": "task_tool_incident",
        "scenario": "payroll",
        "task_idx": 2,
        "tool_name": "create_schedule",
        "arguments": {"name": "Duplicate"},
        "required_for_task": True,
    }
    source.write_text("\n".join([json.dumps(row), json.dumps(row)]) + "\n", encoding="utf-8")

    assert load_incidents(source) == [
        {
            "scenario": "payroll",
            "task_idx": 2,
            "tool_name": "create_schedule",
            "arguments": {"name": "Duplicate"},
            "required_for_task": True,
            "source": str(source),
            "source_kind": "task_tool_incident",
        }
    ]


def test_load_incidents_requires_explicit_arguments_object(tmp_path):
    source = tmp_path / "incidents.jsonl"
    source.write_text(
        json.dumps(
            {
                "kind": "task_tool_incident",
                "scenario": "payroll",
                "task_idx": 2,
                "tool_name": "create_schedule",
                "arguments": "bad",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="arguments"):
        load_incidents(source)
