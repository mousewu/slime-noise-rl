"""Create a reviewable report from bounded AWM verifier-evidence bundles.

The AWM server captures a bundle when a code verifier returns ``others``.
Those bundles are deliberately rich enough for a human to decide whether a
failure came from the policy, a tool/service, or a verifier. This module turns
a directory of bundles into a deterministic report without requiring jq,
OpenEnv, or the AWM server Python environment.

The classifications are observations, not a verdict that a verifier is wrong.
In particular, a database change followed by ``others`` can be either an
incomplete policy action or an over-strict verifier.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_CONTROL_ACTIONS = {"done", "list_tools", "verify", "__list_scenarios__"}
_FALSE_LIKE = {False, 0, "false", "False", "0"}
_INPUT_VALIDATION_ERROR = re.compile(r"^input validation error:\s*", re.IGNORECASE)
_SERVER_ERROR = re.compile(
    r"\b(?:server_error|internal server error|http\s*5\d\d|status(?:\s+code)?\s*5\d\d)\b",
    re.IGNORECASE,
)
_LOG_ERROR_HINT = re.compile(r"\b(?:5\d\d|error|exception|traceback|failed?|timeout)\b", re.IGNORECASE)


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


def _tool_failure_kind(entry: dict[str, Any]) -> str | None:
    """Classify explicit failed calls; never infer failure from normal tool output."""
    reward_type = _string(entry.get("reward_type"))
    error = _string(entry.get("error"))
    if reward_type == "server_error" or _SERVER_ERROR.search(error):
        return "tool_server_error"
    if entry.get("success") not in _FALSE_LIKE:
        return None
    if _INPUT_VALIDATION_ERROR.search(error):
        return "tool_input_validation_error"
    return "tool_execution_error"


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


def _verifier_uses_final_answer(source: dict[str, Any]) -> bool | None:
    """Statically identify a body reference, not merely the common argument."""
    if bool(_field(source, "truncated", False)):
        return None
    text = _field(source, "text")
    if not isinstance(text, str):
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    return any(isinstance(node, ast.Name) and node.id == "final_answer" for node in ast.walk(tree))


def _classify(
    *,
    verifier_result: str,
    execution_status: str,
    observation_error: str,
    entries: list[dict[str, Any]],
    changed_count: int | None,
) -> tuple[str, list[str]]:
    """Assign an evidence-backed primary category and explicit call failure kinds."""
    business_entries = [entry for entry in entries if not _is_control_entry(entry)]
    failure_kinds = sorted({kind for entry in business_entries if (kind := _tool_failure_kind(entry))})
    if execution_status not in {"", "success"}:
        return "verifier_execution_error", failure_kinds
    if observation_error:
        return "verifier_or_server_observation_error", failure_kinds
    if "tool_server_error" in failure_kinds:
        return "tool_server_error", failure_kinds
    if "tool_input_validation_error" in failure_kinds:
        return "tool_input_validation_error", failure_kinds
    if "tool_execution_error" in failure_kinds:
        return "tool_execution_error", failure_kinds
    if verifier_result == "complete":
        return "verified_complete", failure_kinds
    if not business_entries:
        # The adapter converts a parsed model ``done`` action into a hidden
        # ``verify`` call, so this is strong evidence of immediate submission.
        return "submitted_done_without_business_tool", failure_kinds
    if changed_count is None:
        return "business_tool_calls_database_diff_unavailable", failure_kinds
    if changed_count == 0:
        return "business_tool_calls_without_database_change", failure_kinds
    return "database_changed_but_verifier_noncomplete", failure_kinds


def _sample(path: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    entries = _trajectory_entries(evidence)
    business_entries = [entry for entry in entries if not _is_control_entry(entry)]
    changed_count, changed_names = _changed_tables(evidence)
    verifier_result, execution_status, observation_error = _verification(evidence)
    verifier = _field(evidence, "verifier", {})
    source = _field(verifier, "source", {})
    category, failure_kinds = _classify(
        verifier_result=verifier_result,
        execution_status=execution_status,
        observation_error=observation_error,
        entries=entries,
        changed_count=changed_count,
    )
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
                "arguments": entry.get("arguments"),
                "success": entry.get("success"),
                "error": _string(entry.get("error")),
                "failure_kind": _tool_failure_kind(entry),
            }
            for entry in business_entries
        ],
        "tool_failure_kinds": failure_kinds,
        "changed_tables": changed_count,
        "changed_table_names": changed_names,
        "verifier_sha256": _string(_field(source, "sha256")),
        "verifier_truncated": bool(_field(source, "truncated", False)),
        "verifier_uses_final_answer": _verifier_uses_final_answer(source),
        "subprocess_log_error_hint": bool(
            _LOG_ERROR_HINT.search(_string(_field(_field(evidence, "subprocess_log", {}), "tail")))
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


def _write_rows_tsv(path: Path, fields: tuple[str, ...], samples: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\t".join(fields) + "\n")
        for sample in samples:
            stream.write("\t".join(_safe_tsv(sample.get(field)) for field in fields) + "\n")


def _write_auxiliary_reports(output: Path, samples: list[dict[str, Any]]) -> None:
    final_answer_candidates = [
        sample for sample in samples if sample["category"] == "business_tool_calls_without_database_change"
    ]
    _write_rows_tsv(
        output / "final_answer_audit.tsv",
        (
            "scenario",
            "task_idx",
            "category",
            "verifier_uses_final_answer",
            "verifier_truncated",
            "task",
            "evidence_path",
        ),
        final_answer_candidates,
    )
    priority_candidates = sorted(
        (
            sample
            for sample in samples
            if sample["category"] == "database_changed_but_verifier_noncomplete"
        ),
        key=lambda sample: (
            sample["business_tool_call_count"],
            len(sample["changed_table_names"]),
            sample["scenario"],
            str(sample["task_idx"]),
        ),
    )
    _write_rows_tsv(
        output / "priority_verifier_review.tsv",
        (
            "scenario",
            "task_idx",
            "business_tool_call_count",
            "business_tool_calls",
            "changed_table_names",
            "verifier_uses_final_answer",
            "task",
            "evidence_path",
        ),
        priority_candidates,
    )


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
    final_answer = summary["verifier_final_answer_usage"]
    lines.extend(
        [
            "",
            f"- Valid bundles: {summary['valid_bundles']}",
            f"- Invalid JSON bundles: {summary['invalid_bundles']}",
            f"- Unique tasks: {summary['unique_tasks']}",
            f"- Verifiers reading `final_answer`: {final_answer.get('true', 0)}",
            "",
            "## Generated review files",
            "",
            "- `final_answer_audit.tsv`: every no-database-change trajectory, with a static verifier-body check for `final_answer`.",
            "- `priority_verifier_review.tsv`: every database-changed-but-noncomplete trajectory, ordered by fewest tool calls for manual state/verifier comparison.",
            "",
            "## How to interpret categories",
            "",
            "- `submitted_done_without_business_tool`: no task tool call appears before verification. In the current adapter this indicates the model submitted `done` immediately.",
            "- `business_tool_calls_without_database_change`: task tools were called but the captured database state did not change. This can be expected for a read-only task; check `final_answer_audit.tsv` before treating it as a policy failure.",
            "- `database_changed_but_verifier_noncomplete`: a state change occurred but the verifier did not pass; use `priority_verifier_review.tsv` and compare task, tool arguments, database state, and verifier source.",
            "- `tool_input_validation_error`: the model submitted an argument rejected by the tool schema; this is policy/schema-grounding failure, not server failure.",
            "- `tool_server_error`, `tool_execution_error`, and verifier error categories require inspection of the full evidence and server log.",
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
    final_answer_usage = Counter(
        "unknown" if sample["verifier_uses_final_answer"] is None else str(sample["verifier_uses_final_answer"]).lower()
        for sample in samples
    )
    summary = {
        "source": str(source),
        "discovered_evidence_files": len(paths),
        "valid_bundles": len(samples),
        "invalid_bundles": len(invalid),
        "unique_tasks": len({(sample["scenario"], sample["task_idx"]) for sample in samples}),
        "categories": dict(sorted(categories.items())),
        "verifier_results": dict(sorted(Counter(sample["verifier_result"] for sample in samples).items())),
        "verifier_final_answer_usage": dict(sorted(final_answer_usage.items())),
        "changed_tables": dict(sorted(Counter(str(sample["changed_tables"]) for sample in samples).items())),
        "invalid": invalid,
    }
    _write_jsonl(output / "samples.jsonl", samples)
    _write_rows_tsv(
        output / "samples.tsv",
        (
            "category",
            "scenario",
            "task_idx",
            "verifier_result",
            "verify_execution_status",
            "business_tool_call_count",
            "changed_tables",
            "changed_table_names",
            "tool_failure_kinds",
            "verifier_uses_final_answer",
            "observation_error",
            "evidence_path",
        ),
        samples,
    )
    _write_auxiliary_reports(output, samples)
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
