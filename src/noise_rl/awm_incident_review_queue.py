"""Build a conservative review queue from freshly replayed AWM tool incidents.

The fixed-policy pass can identify an action that received a terminal 422/500,
and ``awm_incident_replay`` can reproduce that action in a fresh AWM database.
Neither fact establishes that the action was *required* to complete the task:
the policy may simply have called a duplicate or wrongly ordered tool.  This
module therefore makes review easier without granting itself authority to
exclude data.  Every emitted row retains ``required_for_task: false``.

The queue includes task text from the local AgentWorldModel data and references
the matching full model-replay trace line(s).  A reviewer copies only the
confirmed task-required rows to a separate reviewed-incidents JSONL file and
sets ``required_for_task`` to true.  ``preflight_awm_tasks.sh`` then performs
the final fresh-session check and filters the manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .awm_manifest import normalize_scenario_name


_REPRODUCED_FAILURE_STATUSES = {
    "confirmed_http_422",
    "confirmed_http_500",
    "target_tool_not_discoverable",
}


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append((line_number, row))
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _action_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    scenario, task_idx = row.get("scenario"), row.get("task_idx")
    tool_name, arguments = row.get("tool_name"), row.get("arguments")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("incident has no scenario")
    if type(task_idx) is not int or task_idx < 0:
        raise ValueError(f"incident has invalid task_idx for {scenario!r}")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError(f"incident has no tool_name for {scenario}/{task_idx}")
    if not isinstance(arguments, dict):
        raise ValueError(f"incident arguments must be an object for {scenario}/{task_idx}")
    return scenario, task_idx, tool_name, json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _task_text_index(data_root: Path) -> dict[tuple[str, int], str]:
    path = data_root / "gen_tasks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing AWM task catalog: {path}")
    index: dict[tuple[str, int], str] = {}
    seen_scenarios: set[str] = set()
    for line_number, row in _read_jsonl(path):
        scenario, tasks = row.get("scenario"), row.get("tasks")
        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError(f"Invalid scenario in {path}:{line_number}")
        scenario = normalize_scenario_name(scenario)
        if scenario in seen_scenarios:
            raise ValueError(f"Duplicate normalized scenario {scenario!r} in {path}:{line_number}")
        seen_scenarios.add(scenario)
        if not isinstance(tasks, list) or not all(isinstance(value, str) and value.strip() for value in tasks):
            raise ValueError(f"Invalid task list in {path}:{line_number}")
        index.update({(scenario, task_idx): text for task_idx, text in enumerate(tasks)})
    if not index:
        raise ValueError(f"No tasks found in {path}")
    return index


def _model_trace_references(
    evidence_paths: set[str],
    keys: set[tuple[str, int, str, str]],
) -> tuple[dict[tuple[str, int, str, str], list[dict[str, Any]]], list[dict[str, str]]]:
    """Find physical JSONL line numbers containing each terminal model action."""
    references: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    unavailable: list[dict[str, str]] = []
    for value in sorted(evidence_paths):
        path = Path(value).expanduser()
        if not path.is_file():
            unavailable.append({"model_replay_output": value, "error": "file is unavailable"})
            continue
        try:
            rows = _read_jsonl(path)
        except (OSError, ValueError) as exc:
            unavailable.append({"model_replay_output": value, "error": f"{type(exc).__name__}: {exc}"})
            continue
        for line_number, row in rows:
            task = row.get("task")
            if not isinstance(task, dict):
                continue
            scenario, task_idx = task.get("scenario"), task.get("task_idx")
            if not isinstance(scenario, str) or type(task_idx) is not int:
                continue
            for step_number, step in enumerate(row.get("steps", []), 1):
                if not isinstance(step, dict):
                    continue
                try:
                    action = json.loads(step.get("action", ""))
                except (TypeError, ValueError):
                    continue
                if not isinstance(action, dict):
                    continue
                candidate = {
                    "scenario": scenario,
                    "task_idx": task_idx,
                    "tool_name": action.get("tool_name"),
                    "arguments": action.get("arguments"),
                }
                try:
                    key = _action_key(candidate)
                except ValueError:
                    continue
                if key not in keys:
                    continue
                references[key].append(
                    {
                        "model_replay_output": str(path),
                        "line": line_number,
                        "attempt": row.get("attempt"),
                        "status": row.get("status"),
                        "termination": row.get("termination"),
                        "step": step_number,
                    }
                )
    return dict(references), unavailable


def _compact_response(value: Any) -> Any:
    """Keep the review queue bounded while retaining the observed failure text."""
    if not isinstance(value, dict):
        return value
    selected = {}
    for name in ("reward_type", "error", "warning", "tool_result"):
        if name in value:
            selected[name] = value[name]
    return selected or value


def _write_new_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _write_new_markdown(path: Path, summary: dict[str, Any], queue: list[dict[str, Any]]) -> None:
    lines = [
        "# AWM reproduced-incident review queue",
        "",
        "This is a review queue, not an exclusion decision. Every JSONL row retains `required_for_task: false`; do not use it as `AWM_INCIDENTS` for filtering until a reviewer has established that the exact action is required to complete the task.",
        "",
        "## Summary",
        "",
        f"- Recheck rows read: {summary['recheck_rows']}",
        f"- Distinct actions with a reproducible terminal failure: {summary['queue_rows']}",
        f"- Actions skipped because another recheck recovered or was inconclusive: {summary['inconsistent_action_count']}",
        f"- Missing task-text mappings: {summary['missing_task_text_count']}",
        f"- Model-replay files unavailable while building references: {summary['unavailable_model_replay_file_count']}",
        "",
        "## Decision rule",
        "",
        "For each row, inspect the task text and the referenced full model trace. Copy only rows where the exact tool and arguments are a necessary task step into a separate reviewed-incidents JSONL, change `required_for_task` to `true`, then run `scripts/preflight_awm_tasks.sh`. A duplicate, premature, or otherwise policy-caused action must remain `false`.",
        "",
        "## Reproducible candidates",
        "",
        "| Task | Recheck status | Tool | Trace references |",
        "| --- | --- | --- | ---: |",
    ]
    for row in queue:
        task = f"{row['scenario']}/{row['task_idx']}"
        statuses = ", ".join(row["recheck"]["statuses"])
        lines.append(f"| `{task}` | `{statuses}` | `{row['tool_name']}` | {len(row['model_trace_references'])} |")
    if not queue:
        lines.append("| _none_ | — | — | 0 |")
    lines.extend(
        [
            "",
            "The machine-readable queue contains task text, exact arguments, fresh-session responses, and `path:line` trace references.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def build_review_queue(
    recheck_path: str | Path,
    output_path: str | Path,
    *,
    data_root: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Create non-excluding review rows from fresh-session AWM incident replays."""
    recheck = Path(recheck_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser()
    report = Path(report_path).expanduser()
    if output.resolve() == report.resolve():
        raise ValueError("Review queue JSONL and Markdown report paths must be distinct")
    existing = [path for path in (output, report) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing review artifact: {existing[0]}")
    tasks = _task_text_index(Path(data_root).expanduser().resolve(strict=True))
    recheck_rows = _read_jsonl(recheck)
    grouped: dict[tuple[str, int, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for line_number, row in recheck_rows:
        try:
            key = _action_key(row)
        except ValueError as exc:
            raise ValueError(f"Invalid recheck incident at {recheck}:{line_number}: {exc}") from exc
        grouped[key].append((line_number, row))

    consistent: dict[tuple[str, int, str, str], list[tuple[int, dict[str, Any]]]] = {}
    inconsistent = 0
    for key, rows in grouped.items():
        statuses = {str(row.get("status", "unknown")) for _, row in rows}
        if statuses and statuses.issubset(_REPRODUCED_FAILURE_STATUSES):
            consistent[key] = rows
        else:
            inconsistent += 1

    evidence_paths = {
        str(row["evidence_output"])
        for rows in consistent.values()
        for _, row in rows
        if isinstance(row.get("evidence_output"), str) and row["evidence_output"]
    }
    trace_references, unavailable = _model_trace_references(evidence_paths, set(consistent))
    queue: list[dict[str, Any]] = []
    missing_task_text = 0
    for key, rows in sorted(consistent.items()):
        scenario, task_idx, tool_name, serialized_arguments = key
        first = rows[0][1]
        task_text = tasks.get((scenario, task_idx))
        if task_text is None:
            missing_task_text += 1
        recheck_records = [
            {
                "line": line_number,
                "status": str(row.get("status", "unknown")),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "response": _compact_response(row.get("response")),
            }
            for line_number, row in rows
        ]
        queue.append(
            {
                "kind": "task_tool_incident",
                "scenario": scenario,
                "task_idx": task_idx,
                "task_id": first.get("task_id") or f"awm/{scenario}/{task_idx}",
                "task_text": task_text,
                "tool_name": tool_name,
                "arguments": json.loads(serialized_arguments),
                "required_for_task": False,
                "excluded_from_training": False,
                "source_kind": "fresh_recheck_review_queue",
                "review_decision": "pending",
                "review_note": (
                    "Fresh-session failure reproduced. Inspect task_text and model_trace_references; "
                    "set required_for_task=true only if this exact action is necessary to complete the task."
                ),
                "recheck": {
                    "source": str(recheck),
                    "statuses": sorted({item["status"] for item in recheck_records}),
                    "records": recheck_records,
                },
                "model_trace_references": trace_references.get(key, []),
            }
        )
    summary = {
        "schema": 1,
        "phase": "awm_incident_review_queue",
        "recheck": str(recheck),
        "data_root": str(Path(data_root).expanduser().resolve()),
        "output": str(output.resolve()),
        "report": str(report.resolve()),
        "recheck_rows": len(recheck_rows),
        "recheck_status_counts": dict(sorted(Counter(str(row.get("status", "unknown")) for _, row in recheck_rows).items())),
        "distinct_actions": len(grouped),
        "queue_rows": len(queue),
        "inconsistent_action_count": inconsistent,
        "missing_task_text_count": missing_task_text,
        "unavailable_model_replay_file_count": len(unavailable),
        "unavailable_model_replay_files": unavailable,
        "policy": (
            "All output rows are non-excluding review candidates. A human must establish that the exact "
            "action is required before copying it to a reviewed incidents file with required_for_task=true."
        ),
    }
    _write_new_jsonl(output, queue)
    _write_new_markdown(report, summary, queue)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recheck", required=True, type=Path, help="fresh-session incident replay JSONL")
    parser.add_argument("--data-root", required=True, type=Path, help="local AgentWorldModel-1K directory")
    parser.add_argument("--output", required=True, type=Path, help="new non-excluding review-queue JSONL")
    parser.add_argument("--report", required=True, type=Path, help="new Markdown review report")
    args = parser.parse_args(argv)
    result = build_review_queue(
        args.recheck,
        args.output,
        data_root=args.data_root,
        report_path=args.report,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
