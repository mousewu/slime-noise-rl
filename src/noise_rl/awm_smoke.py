"""Exercise the project AWM harness against a live local AWM server.

This is intentionally not an LLM evaluation and never backpropagates.  It
resets a bounded number of manifest tasks and submits a ``done`` action with a
sentinel final answer.  The run verifies three integration properties against
the real server: final-answer forwarding, normal ``others`` termination, and
clean session shutdown.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .awm import AWMEnvironment, canonical_action
from .data import atomic_json, read_records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_harness_smoke(
    data: Path,
    url: str,
    output: Path,
    *,
    tasks: int,
    final_answer: str,
    timeout: float,
) -> dict[str, Any]:
    """Run a no-write AWM protocol check and persist every attempted session."""
    if output.exists() or Path(str(output) + ".summary.json").exists():
        raise FileExistsError(f"Smoke output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if tasks < 1:
        raise ValueError("tasks must be positive")
    if not final_answer.strip():
        raise ValueError("final_answer must be nonempty so the transport is exercised")
    records = read_records(data)
    awm_tasks = [record["metadata"]["task"] for record in records if record["metadata"]["task"]["environment"] == "awm"]
    if not awm_tasks:
        raise ValueError("Manifest contains no AWM tasks")
    selected = awm_tasks[:tasks]
    action = canonical_action(json.dumps({"tool_name": "done", "arguments": {"final_answer": final_answer}}))
    rows = []
    for task in selected:
        environment = AWMEnvironment(task, url, timeout=timeout)
        row: dict[str, Any] = {
            "task_id": task["id"],
            "scenario": task["scenario"],
            "task_idx": task["task_idx"],
            "final_answer_bytes": len(final_answer.encode("utf-8")),
        }
        try:
            reset = environment.reset()
            result = environment.step(action)
            awm_info = result.info.get("awm", {}) if isinstance(result.info, dict) else {}
            row.update(
                reset_terminated=reset.terminated,
                terminated=result.terminated,
                success=result.success,
                verifier_reward_type=awm_info.get("verifier_reward_type"),
                verify_execution_status=awm_info.get("verify_execution_status"),
                final_answer_submitted=awm_info.get("final_answer_submitted"),
                error=None,
            )
        except Exception as exc:
            row.update(terminated=False, success=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            environment.close()
        rows.append(row)
    _write_jsonl(output, rows)
    errors = [row for row in rows if row.get("error")]
    status_counts = Counter(str(row.get("verifier_reward_type")) for row in rows if not row.get("error"))
    summary = {
        "data": str(data),
        "url": url,
        "tasks_requested": tasks,
        "tasks_run": len(rows),
        "errors": len(errors),
        "terminated": sum(bool(row.get("terminated")) for row in rows),
        "final_answer_submitted": sum(bool(row.get("final_answer_submitted")) for row in rows),
        "verifier_reward_types": dict(sorted(status_counts.items())),
        "output": str(output),
        "pass": not errors
        and all(row.get("terminated") for row in rows)
        and all(row.get("final_answer_submitted") for row in rows),
        "note": "This verifies the harness transport and normal termination only; it is not a policy-quality evaluation.",
    }
    atomic_json(str(output) + ".summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="local AWM manifest")
    parser.add_argument("--url", required=True, help="running local AWM server URL")
    parser.add_argument("--output", required=True, type=Path, help="new JSONL smoke output")
    parser.add_argument("--tasks", type=int, default=8, help="number of manifest tasks to reset and verify")
    parser.add_argument("--final-answer", default="NOISE_RL_AWM_HARNESS_SMOKE", help="nonempty final-answer transport sentinel")
    parser.add_argument("--timeout", type=float, default=180, help="per-session AWM timeout in seconds")
    args = parser.parse_args(argv)
    summary = run_harness_smoke(
        args.data.expanduser(),
        args.url,
        args.output.expanduser(),
        tasks=args.tasks,
        final_answer=args.final_answer,
        timeout=args.timeout,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
