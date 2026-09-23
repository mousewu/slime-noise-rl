"""Filter AWM tasks using repeated, clean teacher replay evidence.

Only infrastructure-stable tasks are admitted to the RL manifest. Successful
replay rows are copied to a separate SFT candidate file; they are still
validated later by ``build_awm_sft_data.sh``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import read_records, write_records


INFRA_FAILURES = {"server_error", "timeout"}
INFRA_EXCEPTION_HINTS = ("ConnectError", "ReadError", "ConnectionClosed", "reset")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def _key(row: dict[str, Any]) -> tuple[str, int]:
    task = row.get("task", {})
    if not isinstance(task, dict) or not isinstance(task.get("scenario"), str):
        raise ValueError("Replay row lacks task.scenario")
    if type(task.get("task_idx")) is not int:
        raise ValueError("Replay row lacks integer task.task_idx")
    return task["scenario"], task["task_idx"]


def _infra_failure(row: dict[str, Any]) -> str | None:
    if row.get("status") == "tool_terminal_failure":
        kind = str(row.get("failure_type", "unknown"))
        if kind in INFRA_FAILURES or kind == "unknown":
            return kind
    if row.get("status") == "episode_exception":
        error = str(row.get("error", ""))
        if any(hint.lower() in error.lower() for hint in INFRA_EXCEPTION_HINTS):
            return "connection_exception"
    return None


def filter_teacher_replay(
    *,
    manifest: Path,
    replay: Path,
    rl_output: Path,
    sft_output: Path,
    report: Path,
    min_attempts: int,
    max_infra_failures: int,
    max_format_errors: int,
    max_non_success: int | None,
    max_tasks_per_scenario: int,
) -> dict[str, Any]:
    records = read_records(manifest)
    replay_rows = _read_jsonl(replay)
    manifest_by_key = {}
    for record in records:
        task = record["metadata"]["task"]
        if task.get("environment") != "awm":
            raise ValueError("Teacher filtering accepts an AWM-only manifest")
        if task.get("split") != "train":
            raise ValueError("Teacher filtering requires a train-only manifest")
        manifest_by_key[(task["scenario"], task["task_idx"])] = record

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        key = _key(row)
        if key in manifest_by_key:
            grouped[key].append(row)

    decisions = {}
    for key, rows in grouped.items():
        infra = Counter(failure for row in rows if (failure := _infra_failure(row)))
        formats = sum(row.get("status") == "format_error" for row in rows)
        non_success = sum(row.get("status") != "success" for row in rows)
        reasons = []
        if len(rows) < min_attempts:
            reasons.append("insufficient_attempts")
        if sum(infra.values()) > max_infra_failures:
            reasons.append("infrastructure_failure")
        if formats > max_format_errors:
            reasons.append("format_error")
        if max_non_success is not None and non_success > max_non_success:
            reasons.append("too_many_non_success")
        decisions[key] = {
            "keep": not reasons,
            "reasons": reasons,
            "attempts": len(rows),
            "successes": sum(row.get("status") == "success" for row in rows),
            "infra_failures": dict(sorted(infra.items())),
            "format_errors": formats,
            "non_success": non_success,
        }

    kept_keys = {key for key, decision in decisions.items() if decision["keep"]}
    missing = set(manifest_by_key) - set(decisions)
    kept_records = [manifest_by_key[key] for key in manifest_by_key if key in kept_keys]
    if not kept_records:
        raise ValueError("No stable tasks remain; no output was written")

    # Cap by scenario deterministically, preserving manifest order.
    counts = Counter()
    capped = []
    capped_keys = set()
    for record in kept_records:
        task = record["metadata"]["task"]
        scenario = task["scenario"]
        if counts[scenario] >= max_tasks_per_scenario:
            continue
        counts[scenario] += 1
        capped.append(record)
        capped_keys.add((scenario, task["task_idx"]))

    sft_rows = []
    for row in replay_rows:
        key = _key(row)
        if key in capped_keys and row.get("status") == "success":
            sft_rows.append(row)

    write_records(rl_output, capped)
    write_records(sft_output, sft_rows)
    result = {
        "schema": 1,
        "manifest": str(manifest),
        "replay": str(replay),
        "rl_output": str(rl_output),
        "sft_replay_output": str(sft_output),
        "tasks_input": len(records),
        "tasks_with_replay": len(grouped),
        "tasks_missing_replay": len(missing),
        "tasks_stable_before_cap": len(kept_keys),
        "tasks_kept_for_rl": len(capped),
        "successful_replay_rows_for_sft": len(sft_rows),
        "excluded_reasons": dict(sorted(Counter(reason for d in decisions.values() for reason in d["reasons"]).items())),
        "policy": {
            "min_attempts": min_attempts,
            "max_infra_failures": max_infra_failures,
            "max_format_errors": max_format_errors,
            "max_non_success": max_non_success,
            "max_tasks_per_scenario": max_tasks_per_scenario,
            "infra_failures": sorted(INFRA_FAILURES | {"connection_exception", "unknown"}),
        },
        "tasks": [
            {"scenario": s, "task_idx": i, **decisions.get((s, i), {"keep": False, "reasons": ["missing_replay"]})}
            for s, i in sorted(manifest_by_key)
        ],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report}")
    result["report"] = str(report)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--rl-output", required=True, type=Path)
    parser.add_argument("--sft-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--min-attempts", type=int, default=4)
    parser.add_argument("--max-infra-failures", type=int, default=0)
    parser.add_argument("--max-format-errors", type=int, default=0)
    parser.add_argument("--max-non-success", type=int, default=None)
    parser.add_argument("--max-tasks-per-scenario", type=int, default=300)
    args = parser.parse_args(argv)
    result = filter_teacher_replay(
        manifest=args.manifest.expanduser().resolve(strict=True),
        replay=args.replay.expanduser().resolve(strict=True),
        rl_output=args.rl_output.expanduser(),
        sft_output=args.sft_output.expanduser(),
        report=args.report.expanduser(),
        min_attempts=args.min_attempts,
        max_infra_failures=args.max_infra_failures,
        max_format_errors=args.max_format_errors,
        max_non_success=args.max_non_success,
        max_tasks_per_scenario=args.max_tasks_per_scenario,
    )
    print(json.dumps({key: result[key] for key in ("tasks_input", "tasks_kept_for_rl", "successful_replay_rows_for_sft", "rl_output", "sft_replay_output", "report")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
