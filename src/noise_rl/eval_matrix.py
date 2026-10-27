"""Merge sharded evaluation outputs and build compact matrix reports."""

import argparse
import csv
import json
from pathlib import Path

from .data import atomic_json, write_records
from .metrics import summarize


def merge_shards(shards, output):
    rows = []
    seen = set()
    for shard in map(Path, shards):
        with shard.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                plan = row["plan"]
                identity = (plan["task_id"], plan["rank"], plan["policy_seed"])
                if identity in seen:
                    raise ValueError(f"Duplicate evaluation episode in {shard}:{line_number}: {identity}")
                seen.add(identity)
                rows.append(row)
    rows.sort(key=lambda row: (row["plan"]["group_id"], row["plan"]["rank"]))
    write_records(output, rows)
    report = summarize(rows)
    atomic_json(str(output) + ".summary.json", report)
    return report


def build_table(root, output_tsv, output_json):
    root = Path(root)
    rows = []
    for path in sorted(root.glob("*/*/episodes.jsonl.summary.json")):
        checkpoint, condition = path.relative_to(root).parts[:2]
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "checkpoint": checkpoint,
                "condition": condition,
                "episodes": value.get("episodes"),
                "tasks": value.get("tasks"),
                "success_rate": value.get("success_rate_task_macro"),
                "ci95_low": value.get("success_rate_task_bootstrap_95ci", [None, None])[0],
                "ci95_high": value.get("success_rate_task_bootstrap_95ci", [None, None])[1],
                "generated_tokens_mean": value.get("mean_generated_tokens"),
                "tool_calls_mean": value.get("mean_tool_calls"),
            }
        )
    if not rows:
        raise ValueError(f"No merged evaluation summaries found under {root}")
    fields = list(rows[0])
    with Path(output_tsv).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output_json, {"schema": 1, "phase": "alfworld_noise_eval_matrix", "rows": rows})
    return {"rows": len(rows), "tsv": str(output_tsv), "json": str(output_json)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--output", required=True)
    merge.add_argument("shards", nargs="+")
    table = commands.add_parser("table")
    table.add_argument("--root", required=True)
    table.add_argument("--output-tsv", required=True)
    table.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    if args.command == "merge":
        result = merge_shards(args.shards, args.output)
    else:
        result = build_table(args.root, args.output_tsv, args.output_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
