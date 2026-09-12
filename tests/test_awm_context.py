import asyncio
import json

import pytest

from noise_rl.awm_context import build_context_checked_manifest, inspect_awm_contexts
from noise_rl.fixtures import ByteTokenizer


def awm_record(task_id, observation):
    scenario, task_idx = task_id.rsplit("/", 1)
    return {
        "prompt": "Complete the tool-use task.",
        "metadata": {
            "task": {
                "id": task_id,
                "environment": "awm",
                "split": "train",
                "scenario": scenario.removeprefix("awm/"),
                "task_idx": int(task_idx),
                "read_only_tools": [],
                "fixture_observation": observation,
            }
        },
    }


async def fixture_observation(task):
    return task["fixture_observation"]


def test_context_inspection_preserves_record_order_and_reports_token_counts():
    records = [
        awm_record("awm/scenario/0", "short"),
        awm_record("awm/scenario/1", "longer observation"),
    ]

    inspections = asyncio.run(
        inspect_awm_contexts(
            records,
            tokenizer=ByteTokenizer(),
            url="http://unused.invalid",
            timeout=1,
            concurrency=2,
            observation_fetcher=fixture_observation,
        )
    )

    assert [item.task_id for item in inspections] == ["awm/scenario/0", "awm/scenario/1"]
    assert all(item.error is None for item in inspections)
    assert inspections[1].prompt_tokens > inspections[0].prompt_tokens


def test_context_checked_manifest_excludes_only_strictly_over_budget_rows(tmp_path):
    records = [
        awm_record("awm/scenario/0", "short"),
        awm_record("awm/scenario/1", "x" * 1000),
    ]
    source = tmp_path / "source.jsonl"
    with source.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    output = tmp_path / "checked.jsonl"
    report = tmp_path / "checked.report.json"

    result = build_context_checked_manifest(
        source,
        output,
        model=None,
        url="http://unused.invalid",
        max_context_tokens=500,
        report_path=report,
        tokenizer=ByteTokenizer(),
        observation_fetcher=fixture_observation,
    )

    assert result["accepted_task_count"] == 1
    assert result["excluded_context_task_count"] == 1
    assert json.loads(output.read_text(encoding="utf-8").strip())["metadata"]["task"]["id"] == "awm/scenario/0"
    assert json.loads(report.read_text(encoding="utf-8"))["acceptance_rule"] == "initial_prompt_tokens < max_context_tokens"


async def broken_observation(_task):
    raise RuntimeError("server unavailable")


def test_context_checked_manifest_fails_without_writing_when_server_has_errors(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(awm_record("awm/scenario/0", "short")) + "\n", encoding="utf-8")
    output = tmp_path / "checked.jsonl"

    with pytest.raises(RuntimeError, match="no output was written"):
        build_context_checked_manifest(
            source,
            output,
            model=None,
            url="http://unused.invalid",
            max_context_tokens=100,
            tokenizer=ByteTokenizer(),
            observation_fetcher=broken_observation,
        )

    assert not output.exists()
