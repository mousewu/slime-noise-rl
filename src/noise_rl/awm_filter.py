"""Exclude AWM scenarios with confirmed broken tools from a manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import read_records, write_records


EXCLUSION_STATUSES = {
    "confirmed_http_422",
    "confirmed_http_500",
    "target_tool_not_discoverable",
    "unconstructible_parameters",
}


def _read_audit(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid audit JSON at {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected audit object at {path}:{number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Audit report has no rows: {path}")
    return rows


def filter_manifest(*, manifest: Path, audit: Path, output: Path, report: Path) -> dict[str, Any]:
    records = read_records(manifest)
    audit_rows = _read_audit(audit)
    reasons: dict[str, set[str]] = defaultdict(set)
    for row in audit_rows:
        scenario = row.get("scenario")
        status = row.get("status")
        if isinstance(scenario, str) and status in EXCLUSION_STATUSES:
            reasons[scenario].add(str(status))
    if not reasons:
        raise ValueError(
            "Audit contains no confirmed 422/500 or parameter-construction failures; refusing to create a misleading filtered manifest"
        )
    kept, removed = [], []
    for record in records:
        task = record["metadata"]["task"]
        if task["environment"] != "awm":
            raise ValueError("AWM tool filtering accepts an AWM-only manifest")
        scenario = task["scenario"]
        if scenario in reasons:
            removed.append({"id": task["id"], "scenario": scenario, "task_idx": task["task_idx"], "reasons": sorted(reasons[scenario])})
        else:
            kept.append(record)
    if not kept:
        raise ValueError("Filtering would remove every training task; no output was written")
    removed_scenarios = {row["scenario"] for row in removed}
    effective_reasons = {scenario: reasons[scenario] for scenario in removed_scenarios}
    write_records(output, kept)
    result = {
        "input_manifest": str(manifest),
        "audit": str(audit),
        "output_manifest": str(output),
        "tasks_input": len(records),
        "tasks_kept": len(kept),
        "tasks_removed": len(removed),
        "scenarios_identified_in_audit": len(reasons),
        "scenarios_removed": len(effective_reasons),
        "status_counts": dict(
            sorted(Counter(reason for values in effective_reasons.values() for reason in values).items())
        ),
        "removed": removed,
        "policy": "exclude every task in a scenario with a confirmed HTTP 422/500, unavailable target tool, or unconstructible tool schema; removing only the tool would leave tasks that require it impossible",
    }
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="input AWM training JSONL")
    parser.add_argument("--audit", required=True, type=Path, help="tool-probe JSONL from noise_rl.awm_tool_audit")
    parser.add_argument("--output", required=True, type=Path, help="new filtered manifest path")
    parser.add_argument("--report", required=True, type=Path, help="new JSON summary of all removals")
    args = parser.parse_args(argv)
    result = filter_manifest(
        manifest=args.manifest.expanduser().resolve(strict=True),
        audit=args.audit.expanduser().resolve(strict=True),
        output=args.output.expanduser(),
        report=args.report.expanduser(),
    )
    print(json.dumps({key: result[key] for key in ("tasks_input", "tasks_kept", "tasks_removed", "scenarios_removed", "output_manifest", "report") if key in result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
