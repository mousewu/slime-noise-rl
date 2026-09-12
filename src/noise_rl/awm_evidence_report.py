"""Create a reviewable report from bounded AWM verifier-evidence bundles.

The AWM server captures a bundle when a code verifier returns ``others``.
Those bundles are deliberately rich enough for a human to decide whether a
failure came from the policy, a tool/service, or a verifier.  This module
turns a directory of bundles into a small, deterministic report without
requiring jq, OpenEnv, or the AWM server Python environment.

The classifications in this report are observations, not a verdict that a
verifier is wrong.  In particular, a database change followed by ``others``
can be either an incomplete policy action or an over-strict verifier.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_CONTROL_ACTIONS = {"done", "list_tools", "verify", "__list_scenarios__"}
_FALSE_LIKE = {False, 0, "false", "False", "0"}
_ERROR_HINT = re.compile(r"\b(?:5\d\d|error|exception|traceback|failed?|timeout)\b", re.IGNORECASE)


def _string(value: Any) -> str:
    """Normalize diagnostic values while treating literal ``None`` as empty."""
    if value is None:
        return ""
    rendered = str(value).strip()
    return "" if rendered.lower() in {"none", "null"} else rendered


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else default


def _evidence_paths(source: Path) -> list[Path]:
    if source.is_file():
        if source.name != "evidence.json":
            raise ValueError(f"Input file must be named evidence.json: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Evidence input does not exist: {source}")
    return sorted(path for path in source.rglob("evidence.json") if path.parent.name.startswith("awm-evidence-"))


def _trajectory_entries(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = _field(evidence, "trajectory", {})
    entries = _field(trajectory, "entries", [])
    return [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


def _entry_tool_name(entry: dict[str, Any]) -> str:
    for name in ("tool_name", "tool", "name"):
        value = _string(entry.get(name))
        if value:
            return value
    return ""


def _is_control_entry(entry: dict[str, Any]) -> bool:
    return _string(entry.get("action")) in _CONTROL_ACTIONS or _entry_tool_name(entry) in _CONTROL_ACTIONS


def _entry_error(entry: dict[str, Any]) -> bool:
    if entry.get("success") in _FALSE_LIKE:
        return True
    text = " ".join(
        _string(entry.get(name)) for name in ("error", "observation", "result", "reward_type")
    )
    return bool(_ERROR_HINT.search(text))


def _changed_tables(evidence: dict[str, Any]) -> tuple[int | None, list[str]]:
    database = _field(evidence, "database", {})
    diff = _field(database, "diff", {})
    changed_count = _field(diff, "changed_tables")
    if not isinstance(changed_count, int):
        changed_count = None
    tables = _field(diff, "tables", [])
    names = []
    if isinstance(tables, list):
        names = sorted(
            _string(table.get("table"))
            for table in tables
            if isinstance(table, dict) and table.get("changed") and _string(table.get("table"))
        )
    return changed_count, names


def _verification(evidence: dict[str, Any]) -> tuple[str, str, str]:
    observation = _field(evidence, "observation", {})
    verify = _field(observation, "verify_result", {})
    raw = _field(verify, "raw_result", {})
    result = _string(_field(verify, "result")) or _string(_field(raw, "result"))
    return result, _string(_field(verify, "execution_status")), _string(_field(observation, "error"))


def _classify(
    *, verifier_result: str, execution_status: str, observation_error: str, entries: list[dict[str, Any]], changed_count: int | None
) -> str:
    """Assign only an evidence-backed, mutually exclusive primary category."""
    business_entries = [entry for entry in entries if not _is_control_entry(entry)]
    has_error = bool(_ERROR_HINT.search(observation_error)) or execution_status not in {"", "success"}
    has_error = has_error or any(_entry_error(entry) for entry in business_entries)
    if has_error:
        return "tool_or_service_error"
    if verifier_result == "complete":
        return "verified_complete"
    if not business_entries:
        # AWM's adapter turns the model's parsed ``done`` action into its hidden
        # ``verify`` action, so this is the strongest available evidence of an
        # immediate submission without a task tool call.
        return "submitted_done_without_business_tool"
    if changed_count is None:
        return "business_tool_calls_database_diff_unavailable"
    if changed_count == 0:
        return "business_tool_calls_without_database_change"
    return "database_changed_but_verifier_noncomplete"


def _sample(path: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    entries = _trajectory_entries(evidence)
    business_entries = [entry for entry in entries if not _is_control_entry(entry)]
    changed_count, changed_names = _changed_tables(evidence)
    verifier_result, execution_status, observation_error = _verification(evidence)
    category = _classify(
        verifier_result=verifier_result,
        execution_status=execution_status,
        observation_error=observation_error,
        entries=entries,
        changed_count=changed_count,
    )
    verifier = _field(evidence, "verifier", {})
    source = _field(verifier, "source", {})
    task = _field(evidence, "task", {})
    return {
        "evidence_path": str(path),
        "scenario": _string(evidence.get("scenario")),
        "task_idx": evidence.get("task_idx"),
        "task": _string(_field(task, "text")),
        "task_truncated": bool(_field(task, "truncated", False)),
        "category": category,
        "verifier_result": verifier_result,
        "verify_execution_status": execution_status,
        "observation_error": observation_error,
        "trajectory_entry_count": len(entries),
        "trajectory_truncated": bool(_field(_field(evidence, "trajectory", {}), "truncated", False)),
        "business_tool_call_count": len(business_entries),
        "business_tool_calls": [
            {
                "action": _string(entry.get("action")),
                "tool_name": _entry_tool_name(entry),
                "success": entry.get("success"),
                "error": _string(entry.get("error")),
            }
            for entry in business_entries
        ],
        "changed_tables": changed_count,
        "changed_table_names": changed_names,
        "verifier_sha256": _string(_field(source, "sha256")),
        "verifier_truncated": bool(_field(source, "truncated", False)),
        "subprocess_log_error_hint": bool(
            _ERROR_HINT.search(_string(_field(_field(evidence, "subprocess_log", {}), "tail")))
        ),
    }


def _safe_tsv(value: Any) -> str:
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _string(value).replace("\t", " ").replace("\n", " ")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")


def _write_tsv(path: Path, samples: list[dict[str, Any]]) -> None:
    fields = (
        "category",
        "scenario",
        "task_idx",
        "verifier_result",
        "verify_execution_status",
        "business_tool_call_count",
        "changed_tables",
        "changed_table_names",
        "observation_error",
        "evidence_path",
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\t".join(fields) + "\n")
        for sample in samples:
            stream.write("\t".join(_safe_tsv(sample.get(field)) for field in fields) + "\n")


def _write_markdown(path: Path, summary: dict[str, Any], samples: list[dict[str, Any]], max_examples: int) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["category"]].append(sample)
    lines = [
        "# AWM verifier evidence report",
        "",
        "This report is descriptive. `others` means the code verifier did not pass; it does not by itself prove a verifier or environment bug.",
        "",
        "## Summary",
        "",
        "| Category | Bundles |",
        "| --- | ---: |",
    ]
    for category, count in summary["categories"].items():
        lines.append(f"| `{category}` | {count} |")
    lines.extend(
        [
            "",
            f"- Valid bundles: {summary['valid_bundles']}",
            f"- Invalid JSON bundles: {summary['invalid_bundles']}",
            f"- Unique tasks: {summary['unique_tasks']}",
            "",
            "## How to interpret categories",
            "",
            "- `submitted_done_without_business_tool`: no task tool call appears before verification. In the current adapter this indicates the model submitted `done` immediately.",
            "- `business_tool_calls_without_database_change`: task tools were called but the captured database state did not change; inspect tool results and arguments.",
            "- `database_changed_but_verifier_noncomplete`: a state change occurred but the verifier did not pass; compare the final state with task requirements and verifier source.",
            "- `tool_or_service_error`: evidence contains a failed tool call, non-success verifier execution, or an explicit error hint; inspect the full bundle and server log.",
            "",
            "## Representative samples",
            "",
        ]
    )
    for category in sorted(grouped):
        lines.extend([f"### `{category}`", ""])
        for sample in grouped[category][:max_examples]:
            task_ref = f"{sample['scenario']}/{sample['task_idx']}"
            tools = ", ".join(
                call["tool_name"] or call["action"] or "unknown" for call in sample["business_tool_calls"]
            ) or "none"
            changes = ", ".join(sample["changed_table_names"]) or "none"
            lines.extend(
                [
                    f"- **{task_ref}** — task tools: `{tools}`; changed tables: `{changes}`; verifier: `{sample['verifier_result'] or 'missing'}`.",
                    f"  - Task: {sample['task'] or '(task text unavailable)'}",
                    f"  - Evidence: `{sample['evidence_path']}`",
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(source: Path, output: Path, *, max_examples: int = 5) -> dict[str, Any]:
    """Read evidence under ``source`` and atomically claim a new output directory."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report directory: {output}")
    if max_examples < 1:
        raise ValueError("max_examples must be at least 1")
    paths = _evidence_paths(source)
    output.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("top-level JSON value is not an object")
            samples.append(_sample(path, value))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid.append({"evidence_path": str(path), "error": str(exc)})
    samples.sort(key=lambda item: (item["scenario"], str(item["task_idx"]), item["evidence_path"]))
    categories = Counter(sample["category"] for sample in samples)
    summary = {
        "source": str(source),
        "discovered_evidence_files": len(paths),
        "valid_bundles": len(samples),
        "invalid_bundles": len(invalid),
        "unique_tasks": len({(sample["scenario"], sample["task_idx"]) for sample in samples}),
        "categories": dict(sorted(categories.items())),
        "verifier_results": dict(sorted(Counter(sample["verifier_result"] for sample in samples).items())),
        "changed_tables": dict(sorted(Counter(str(sample["changed_tables"]) for sample in samples).items())),
        "invalid": invalid,
    }
    _write_jsonl(output / "samples.jsonl", samples)
    _write_tsv(output / "samples.tsv", samples)
    _write_json(output / "summary.json", summary)
    _write_markdown(output / "REPORT.md", summary, samples, max_examples)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="AWM diagnostics directory or one evidence.json")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for REPORT.md and machine-readable rows")
    parser.add_argument("--max-examples", type=int, default=5, help="representative rows per category in REPORT.md")
    args = parser.parse_args(argv)
    summary = build_report(args.input.expanduser(), args.output_dir.expanduser(), max_examples=args.max_examples)
    print(
        "AWM evidence report: "
        f"valid={summary['valid_bundles']} invalid={summary['invalid_bundles']} "
        f"unique_tasks={summary['unique_tasks']} output={args.output_dir}"
    )
    for category, count in summary["categories"].items():
        print(f"  {category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
