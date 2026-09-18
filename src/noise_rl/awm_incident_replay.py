"""Replay observed AWM tool failures in fresh sessions before the next RL run.

This is deliberately narrower than a model rollout.  A training-time AWM
diagnostic records the exact ``scenario``, ``task_idx``, business-tool name and
arguments that reached the server.  Replaying that action after a fresh reset
answers a useful, reproducible question: does the same task/database state
still produce HTTP 422/500?  It never reuses a training database and it never
starts a model, Ray, Slime, or a gradient update.

An incident must be explicitly marked ``required_for_task: true`` before a
reproduced error excludes a training sample.  A policy can call a conflicting
``create_*`` tool in an otherwise solvable task; excluding such a task merely
hides an agent error.  Those incidents are retained as evidence for the
runtime terminal-failure guard, but do not change the manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .awm_tool_audit import STATUS_PATTERN, _compact, _observation_details


_EXCLUDABLE_STATUSES = {"confirmed_http_422", "confirmed_http_500", "target_tool_not_discoverable"}


def _read_json(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid incident JSON at {path}: {exc}") from exc
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError(f"Incident JSON must be an object or object array: {path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid incident JSONL at {path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Incident JSONL row must be an object: {path}:{number}")
        rows.append(value)
    return rows


def _raw_incident_rows(source: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    if source.is_dir():
        files = sorted(source.rglob("awm-diagnostic-*.json"))
        if not files:
            raise ValueError(f"No awm-diagnostic-*.json files found below {source}")
        for path in files:
            for row in _read_json(path):
                yield path, row
        return
    if not source.is_file():
        raise FileNotFoundError(f"Incident source does not exist: {source}")
    loader = _read_json if source.suffix.lower() == ".json" else _read_jsonl
    for row in loader(source):
        yield source, row


def load_incidents(source: Path) -> list[dict[str, Any]]:
    """Load, validate and deduplicate tool-error incidents.

    Server diagnostic files include many possible event kinds.  Only an
    explicit ``tool_server_error`` record is admitted from those files; manual
    JSONL input may use ``kind: task_tool_incident`` for an audited task.
    """
    incidents: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for origin, row in _raw_incident_rows(source):
        kind = row.get("kind")
        if kind not in {"tool_server_error", "task_tool_incident"}:
            continue
        scenario, task_idx = row.get("scenario"), row.get("task_idx")
        tool_name, arguments = row.get("tool_name"), row.get("arguments")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError(f"Incident has no scenario: {origin}")
        if not isinstance(task_idx, int) or task_idx < 0:
            raise ValueError(f"Incident has invalid task_idx for {scenario!r}: {origin}")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"Incident has no tool_name for {scenario}/{task_idx}: {origin}")
        if not isinstance(arguments, dict):
            raise ValueError(f"Incident arguments must be an object for {scenario}/{task_idx}: {origin}")
        required = row.get("required_for_task", False)
        if not isinstance(required, bool):
            raise ValueError(f"required_for_task must be boolean for {scenario}/{task_idx}: {origin}")
        key = (scenario, task_idx, tool_name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        incidents.append(
            {
                "scenario": scenario,
                "task_idx": task_idx,
                "tool_name": tool_name,
                "arguments": arguments,
                "required_for_task": required,
                "source": str(origin),
                "source_kind": kind,
            }
        )
    if not incidents:
        raise ValueError(f"No usable tool-error incidents in {source}")
    return incidents


def _classify(observation: Any) -> str:
    error = str(getattr(observation, "error", "") or "")
    match = STATUS_PATTERN.search(error)
    if match:
        return f"confirmed_http_{match.group(1)}"
    if getattr(observation, "reward_type", None) == "server_error":
        return "unclassified_server_error"
    return "replay_recovered"


async def _replay_one(
    incident: dict[str, Any],
    *,
    url: str,
    connect_timeout: float,
    message_timeout: float,
) -> dict[str, Any]:
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

    row: dict[str, Any] = {"kind": "task_tool_incident_replay", **incident, "trace": []}
    started = time.perf_counter()

    def record(action: str, **details: Any) -> None:
        row["trace"].append({"step": len(row["trace"]) + 1, "action": action, **details})

    try:
        async with AWMEnv(
            base_url=url,
            connect_timeout_s=connect_timeout,
            message_timeout_s=message_timeout,
        ) as env:
            try:
                reset = await env.reset(scenario=incident["scenario"], task_idx=incident["task_idx"])
                record(
                    "reset",
                    arguments={"scenario": incident["scenario"], "task_idx": incident["task_idx"]},
                    response=_observation_details(reset.observation),
                )
                if getattr(reset.observation, "reward_type", None) != "reset_ok":
                    row["status"] = "reset_failed"
                else:
                    listed = await env.step(ListToolsAction())
                    tools = {
                        str(getattr(tool, "name", ""))
                        for tool in getattr(listed.observation, "tools", []) or []
                    }
                    record("list_tools", tool_names=sorted(tools), response=_observation_details(listed.observation))
                    if incident["tool_name"] not in tools:
                        row["status"] = "target_tool_not_discoverable"
                    else:
                        result = await env.step(
                            CallToolAction(tool_name=incident["tool_name"], arguments=incident["arguments"])
                        )
                        record(
                            "call_tool",
                            tool_name=incident["tool_name"],
                            arguments=incident["arguments"],
                            response=_observation_details(result.observation),
                            done=bool(result.done),
                        )
                        row["status"] = _classify(result.observation)
                        row["response"] = _observation_details(result.observation)
            finally:
                try:
                    done = await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))
                    record(
                        "done",
                        arguments={"keep_session": False},
                        response=_observation_details(done.observation),
                        done=bool(done.done),
                    )
                except Exception as exc:
                    record("done", arguments={"keep_session": False}, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        row["status"] = "replay_exception"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["elapsed_seconds"] = round(time.perf_counter() - started, 6)

    status = row.get("status")
    row["excluded_from_training"] = bool(
        incident["required_for_task"] and status in _EXCLUDABLE_STATUSES
    )
    row["exclusion_scope"] = "task"
    if status in _EXCLUDABLE_STATUSES and not incident["required_for_task"]:
        row["retained_reason"] = "replayed action is not marked required_for_task; retain task and rely on terminal-failure handling"
    return row


async def replay_incidents(
    incidents: list[dict[str, Any]],
    *,
    url: str,
    workers: int,
    connect_timeout: float,
    message_timeout: float,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(workers)

    async def bounded(incident: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await _replay_one(
                incident,
                url=url,
                connect_timeout=connect_timeout,
                message_timeout=message_timeout,
            )

    rows: list[dict[str, Any]] = []
    for index in range(0, len(incidents), workers):
        rows.extend(await asyncio.gather(*(bounded(value) for value in incidents[index : index + workers])))
        excluded = sum(bool(row.get("excluded_from_training")) for row in rows)
        print(f"incident replay: {len(rows)}/{len(incidents)}; task exclusions={excluded}", file=sys.stderr, flush=True)
    return rows


def _write_new_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite replay output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite replay summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incidents", required=True, type=Path, help="diagnostic JSON/JSONL file or diagnostics directory")
    parser.add_argument("--url", required=True, help="already-running local AWM server URL")
    parser.add_argument("--output", required=True, type=Path, help="new JSONL replay evidence output")
    parser.add_argument("--workers", type=int, default=2, help="bounded concurrent fresh AWM sessions")
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--message-timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.connect_timeout <= 0 or args.message_timeout <= 0:
        raise ValueError("timeouts must be positive")
    output = args.output.expanduser()
    summary_path = Path(str(output) + ".summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite existing incident replay output or summary")
    incidents = load_incidents(args.incidents.expanduser().resolve(strict=True))
    started = time.perf_counter()
    rows = asyncio.run(
        replay_incidents(
            incidents,
            url=args.url,
            workers=args.workers,
            connect_timeout=args.connect_timeout,
            message_timeout=args.message_timeout,
        )
    )
    counts = Counter(str(row.get("status", "unknown")) for row in rows)
    exclusions = [row for row in rows if row.get("excluded_from_training")]
    summary = {
        "kind": "awm_task_tool_incident_replay",
        "incidents": str(args.incidents),
        "url": args.url,
        "workers": args.workers,
        "incidents_loaded": len(incidents),
        "incidents_replayed": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "tasks_to_exclude": [
            {"scenario": row["scenario"], "task_idx": row["task_idx"], "status": row["status"]}
            for row in exclusions
        ],
        "output": str(output),
        "summary": str(summary_path),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "note": "Only reproduced failures explicitly marked required_for_task exclude the exact task. Other reproduced failures remain runtime-harness evidence.",
    }
    _write_new_jsonl(output, rows)
    _write_new_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
