import json

import pytest

from noise_rl.awm_evidence_report import build_report


def _write_bundle(root, name, *, entries, changed_tables, result="others", execution_status="success"):
    bundle = root / name
    bundle.mkdir(parents=True)
    (bundle / "evidence.json").write_text(
        json.dumps(
            {
                "scenario": "finance",
                "task_idx": int(name.rsplit("-", 1)[-1]),
                "task": {"text": "Create a checking account.", "truncated": False},
                "observation": {
                    "error": None,
                    "verify_result": {
                        "execution_status": execution_status,
                        "raw_result": {"result": result},
                        "result": result,
                    },
                },
                "trajectory": {"entries": entries, "truncated": False},
                "verifier": {"source": {"sha256": "abc", "truncated": False}},
                "database": {
                    "diff": {
                        "changed_tables": changed_tables,
                        "tables": [{"table": "accounts", "changed": bool(changed_tables)}],
                    }
                },
                "subprocess_log": {"tail": ""},
            }
        ),
        encoding="utf-8",
    )


def test_evidence_report_classifies_observable_outcomes_and_refuses_overwrite(tmp_path):
    root = tmp_path / "diagnostics"
    _write_bundle(
        root,
        "awm-evidence-0",
        entries=[{"action": "list_tools", "success": True}, {"action": "verify", "reward_type": "others"}],
        changed_tables=0,
    )
    _write_bundle(
        root,
        "awm-evidence-1",
        entries=[{"action": "call_tool", "tool_name": "create_account", "success": True}],
        changed_tables=0,
    )
    _write_bundle(
        root,
        "awm-evidence-2",
        entries=[{"action": "call_tool", "tool_name": "create_account", "success": True}],
        changed_tables=1,
    )
    _write_bundle(
        root,
        "awm-evidence-3",
        entries=[{"action": "call_tool", "tool_name": "create_account", "success": False, "error": "HTTP 500"}],
        changed_tables=0,
    )

    output = tmp_path / "report"
    summary = build_report(root, output, max_examples=2)

    assert summary["valid_bundles"] == 4
    assert summary["categories"] == {
        "business_tool_calls_without_database_change": 1,
        "database_changed_but_verifier_noncomplete": 1,
        "submitted_done_without_business_tool": 1,
        "tool_or_service_error": 1,
    }
    rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert rows[0]["category"] == "submitted_done_without_business_tool"
    assert rows[2]["changed_table_names"] == ["accounts"]
    assert "does not by itself prove a verifier or environment bug" in (output / "REPORT.md").read_text()
    assert (output / "samples.tsv").read_text().startswith("category\tscenario")
    with pytest.raises(FileExistsError):
        build_report(root, output)
