import json

from noise_rl.awm_incident_review_queue import build_review_queue


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_review_queue_contains_only_consistently_reproduced_actions_and_trace_locations(tmp_path):
    data_root = tmp_path / "awm-data"
    data_root.mkdir()
    _write_jsonl(
        data_root / "gen_tasks.jsonl",
        [{"scenario": "Personal Finance", "tasks": ["Create the required checking account."]}],
    )
    trace = tmp_path / "model-replay.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "task": {"scenario": "personal_finance", "task_idx": 0},
                "attempt": 0,
                "status": "tool_terminal_failure",
                "termination": "environment_terminal",
                "steps": [
                    {
                        "action": json.dumps(
                            {"tool_name": "create_account", "arguments": {"name": "Everyday"}}
                        )
                    }
                ],
            }
        ],
    )
    recheck = tmp_path / "recheck.jsonl"
    _write_jsonl(
        recheck,
        [
            {
                "kind": "task_tool_incident_replay",
                "scenario": "personal_finance",
                "task_idx": 0,
                "task_id": "awm/personal_finance/0",
                "tool_name": "create_account",
                "arguments": {"name": "Everyday"},
                "evidence_output": str(trace),
                "status": "confirmed_http_500",
                "response": {"reward_type": "server_error", "error": "HTTP 500", "unneeded": "drop"},
            },
            {
                "kind": "task_tool_incident_replay",
                "scenario": "personal_finance",
                "task_idx": 0,
                "tool_name": "create_account",
                "arguments": {"name": "Recovered"},
                "status": "replay_recovered",
            },
        ],
    )
    output = tmp_path / "queue.jsonl"
    report = tmp_path / "queue.md"

    summary = build_review_queue(recheck, output, data_root=data_root, report_path=report)

    queue = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["recheck_rows"] == 2
    assert summary["queue_rows"] == 1
    assert summary["inconsistent_action_count"] == 1
    assert queue == [
        {
            "arguments": {"name": "Everyday"},
            "excluded_from_training": False,
            "kind": "task_tool_incident",
            "model_trace_references": [
                {
                    "attempt": 0,
                    "line": 1,
                    "model_replay_output": str(trace),
                    "status": "tool_terminal_failure",
                    "step": 1,
                    "termination": "environment_terminal",
                }
            ],
            "recheck": {
                "records": [
                    {
                        "elapsed_seconds": None,
                        "line": 1,
                        "response": {"error": "HTTP 500", "reward_type": "server_error"},
                        "status": "confirmed_http_500",
                    }
                ],
                "source": str(recheck.resolve()),
                "statuses": ["confirmed_http_500"],
            },
            "required_for_task": False,
            "review_decision": "pending",
            "review_note": (
                "Fresh-session failure reproduced. Inspect task_text and model_trace_references; "
                "set required_for_task=true only if this exact action is necessary to complete the task."
            ),
            "scenario": "personal_finance",
            "source_kind": "fresh_recheck_review_queue",
            "task_id": "awm/personal_finance/0",
            "task_idx": 0,
            "task_text": "Create the required checking account.",
            "tool_name": "create_account",
        }
    ]
    markdown = report.read_text(encoding="utf-8")
    assert "Distinct actions with a reproducible terminal failure: 1" in markdown
    assert "`personal_finance/0`" in markdown
    assert "not an exclusion decision" in markdown
