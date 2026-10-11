"""Summarize multi-checkpoint AWM replay results by leakage-audited split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .data import atomic_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _bootstrap_ci(values: list[float], seed: int, samples: int = 4000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]]


def summarize_rows(rows: list[dict[str, Any]], task_ids: set[str], seed: int) -> dict[str, Any]:
    selected = [row for row in rows if row.get("task", {}).get("id") in task_ids]
    by_task: dict[str, list[float]] = defaultdict(list)
    statuses: Counter[str] = Counter()
    for row in selected:
        task_id = row["task"]["id"]
        by_task[task_id].append(float(bool(row.get("success"))))
        statuses[str(row.get("status", "unknown"))] += 1
    missing = task_ids - set(by_task)
    if missing:
        raise ValueError(f"Replay is missing {len(missing)} expected tasks; first={sorted(missing)[0]}")
    task_rates = [mean(values) for values in by_task.values()]
    attempts = len(selected)
    successes = sum(sum(values) for values in by_task.values())
    environment_errors = statuses["episode_exception"] + statuses["tool_terminal_failure"]
    return {
        "tasks": len(by_task),
        "attempts": attempts,
        "successes": int(successes),
        "success_rate_micro": successes / attempts if attempts else 0.0,
        "success_rate_task_macro": mean(task_rates) if task_rates else 0.0,
        "task_macro_bootstrap_95_ci": _bootstrap_ci(task_rates, seed),
        "environment_error_rate": environment_errors / attempts if attempts else 0.0,
        "format_error_rate": statuses["format_error"] / attempts if attempts else 0.0,
        "status_counts": dict(sorted(statuses.items())),
    }


def summarize_checkpoint_evaluations(
    protocol: str | Path,
    checkpoint_replays: list[tuple[str, str | Path]],
    output_json: str | Path,
    output_tsv: str | Path,
    seed: int = 20261011,
) -> dict[str, Any]:
    protocol_path = Path(protocol).expanduser().resolve(strict=True)
    definition = json.loads(protocol_path.read_text(encoding="utf-8"))
    set_ids = {
        name: set(definition["sets"][name]["task_ids"])
        for name in ("environment_non_rl", "scenario_heldout", "rl_resample")
    }
    results: dict[str, Any] = {}
    for index, (label, replay) in enumerate(checkpoint_replays):
        replay_path = Path(replay).expanduser().resolve(strict=True)
        rows = _read_jsonl(replay_path)
        results[label] = {
            "replay": str(replay_path),
            "sets": {
                name: summarize_rows(rows, ids, seed + index * 10 + offset)
                for offset, (name, ids) in enumerate(set_ids.items())
            },
        }
    report = {
        "schema": 1,
        "phase": "awm_checkpoint_eval_summary",
        "protocol": str(protocol_path),
        "checkpoints": results,
    }
    output_json = Path(output_json)
    output_tsv = Path(output_tsv)
    if output_json.exists() or output_tsv.exists():
        raise FileExistsError("Refusing to overwrite checkpoint evaluation summary")
    atomic_json(output_json, report)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow([
            "checkpoint", "set", "tasks", "attempts", "success_rate_task_macro",
            "ci_low", "ci_high", "environment_error_rate", "format_error_rate",
        ])
        for label, checkpoint in results.items():
            for name, metrics in checkpoint["sets"].items():
                writer.writerow([
                    label, name, metrics["tasks"], metrics["attempts"],
                    metrics["success_rate_task_macro"],
                    *metrics["task_macro_bootstrap_95_ci"],
                    metrics["environment_error_rate"], metrics["format_error_rate"],
                ])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("LABEL", "REPLAY"), required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args(argv)
    report = summarize_checkpoint_evaluations(
        args.protocol, [(a, b) for a, b in args.checkpoint],
        args.output_json, args.output_tsv, args.seed,
    )
    print(json.dumps({label: value["sets"] for label, value in report["checkpoints"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
