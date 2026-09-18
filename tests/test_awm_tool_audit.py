import json

from noise_rl.awm_filter import filter_manifest
from noise_rl.awm_tool_audit import build_minimal_arguments, find_route_targets
from noise_rl.data import read_records


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _record(scenario, task_idx):
    return {
        "prompt": "Complete the tool-use task.",
        "metadata": {
            "task": {
                "id": f"awm/{scenario}/{task_idx}",
                "environment": "awm",
                "split": "train",
                "scenario": scenario,
                "task_idx": task_idx,
                "read_only_tools": [],
            }
        },
    }


def test_route_audit_uses_operation_id_and_detects_typed_shadow(tmp_path):
    data = tmp_path / "awm"
    data.mkdir()
    code = '''
from fastapi import FastAPI
app = FastAPI()

@app.get("/api/pins/{pin_id}")
def get_pin(pin_id: int):
    return {"id": pin_id}

@app.get("/api/pins/search", operation_id="search_pins")
def search_pins(query: str):
    return {"query": query}
'''
    _write_jsonl(data / "gen_envs.jsonl", [{"scenario": "social_media_2", "full_code": code}])

    targets, summary = find_route_targets(data)

    assert summary["route_shadow_targets"] == 1
    assert targets[0]["tool_name_candidates"] == ["search_pins"]
    assert targets[0]["likely_http_422"] is True
    assert targets[0]["captured_path_parameters"] == [{"name": "pin_id", "annotation": "int"}]


def test_minimal_argument_builder_resolves_definitions_and_required_values():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 3},
            "filters": {"$ref": "#/$defs/filters"},
        },
        "required": ["query", "filters"],
        "$defs": {
            "filters": {
                "type": "object",
                "properties": {"page": {"type": "integer", "minimum": 2}},
                "required": ["page"],
            }
        },
    }

    assert build_minimal_arguments(schema) == {"query": "xxx", "filters": {"page": 2}}


def test_filter_removes_entire_scenario_for_confirmed_tool_failure(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(manifest, [_record("healthy", 0), _record("broken", 0), _record("broken", 1)])
    audit = tmp_path / "tool-audit.jsonl"
    _write_jsonl(
        audit,
        [
            {"scenario": "healthy", "status": "call_completed", "excluded_from_training": False},
            {"scenario": "broken", "status": "confirmed_http_422", "excluded_from_training": True},
        ],
    )
    output = tmp_path / "filtered.jsonl"
    report = tmp_path / "filtered.report.json"

    result = filter_manifest(manifest=manifest, audit=audit, output=output, report=report)

    assert result["tasks_kept"] == 1
    assert result["tasks_removed"] == 2
    assert result["scenarios_identified_in_audit"] == result["scenarios_removed"] == 1
    assert [row["metadata"]["task"]["scenario"] for row in read_records(output)] == ["healthy"]
    assert json.loads(report.read_text(encoding="utf-8"))["status_counts"] == {"confirmed_http_422": 1}


def test_filter_removes_only_task_for_task_scoped_incident_replay(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(manifest, [_record("shared", 0), _record("shared", 1), _record("healthy", 0)])
    audit = tmp_path / "incident-replay.jsonl"
    _write_jsonl(
        audit,
        [
            {
                "scenario": "shared",
                "task_idx": 1,
                "status": "confirmed_http_500",
                "excluded_from_training": True,
                "exclusion_scope": "task",
            },
            {
                "scenario": "shared",
                "task_idx": 0,
                "status": "confirmed_http_500",
                "excluded_from_training": False,
                "exclusion_scope": "task",
            },
        ],
    )
    output = tmp_path / "filtered.jsonl"
    report = tmp_path / "filtered.report.json"

    result = filter_manifest(manifest=manifest, audit=audit, output=output, report=report)

    assert result["tasks_kept"] == 2
    assert result["tasks_removed"] == 1
    assert result["tasks_identified_in_audit"] == 1
    assert [(row["metadata"]["task"]["scenario"], row["metadata"]["task"]["task_idx"]) for row in read_records(output)] == [
        ("shared", 0),
        ("healthy", 0),
    ]
