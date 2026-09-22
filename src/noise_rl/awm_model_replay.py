"""Run a fixed SGLang policy through every selected AWM task without training.

This is a *candidate discovery* pass for task-specific AWM failures.  Each
attempt receives a fresh AWM session/database, uses the production multi-turn
agent harness, and writes the complete generated action trace.  The model is
never updated, noise injection is disabled, and no Ray/Slime training process
is started.

An observed 5xx is not automatically proof that a task is broken: a policy can
send an invalid, duplicate, or poorly ordered business action.  Such calls are
therefore emitted as ``task_tool_incident`` candidates with
``required_for_task: false``.  Review a candidate and mark that field true only
when the failing action is a necessary task step; the existing incident replay
preflight can then reproduce it in a fresh session before filtering a task.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .agent import SGLangClient, Trajectory, run_episode
from .awm_context import load_local_tokenizer
from .config import ExperimentConfig, NoiseConfig, load_config
from .data import atomic_json, read_records
from .metrics import episode_record
from .sampling import plan_sample, stable_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_positive(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _selected_records(
    records: list[dict[str, Any]], *, shard_count: int, shard_index: int, tasks: int
) -> list[dict[str, Any]]:
    """Select a stable manifest shard, then an optional bounded prefix."""
    _require_positive("shard_count", shard_count)
    _require_nonnegative("shard_index", shard_index)
    _require_nonnegative("tasks", tasks)
    if shard_index >= shard_count:
        raise ValueError("shard_index must be smaller than shard_count")
    selected = [
        record
        for record in records
        if stable_seed("awm-model-replay-shard", record["metadata"]["task"]["id"])
        % shard_count
        == shard_index
    ]
    if tasks:
        selected = selected[:tasks]
    if not selected:
        raise ValueError("Selected AWM replay shard has no tasks")
    return selected


def _replay_config(
    config: ExperimentConfig,
    *,
    url: str,
    seed: int,
    concurrency: int,
    max_turns: int | None,
    max_tool_calls: int | None,
    max_context_tokens: int | None,
    max_generated_tokens: int | None,
    max_tokens_per_turn: int | None,
) -> ExperimentConfig:
    """Disable stochastic faults while preserving the real harness budgets."""
    _require_positive("concurrency", concurrency)
    overrides: dict[str, Any] = {
        "awm_url": url,
        "eval_seed": seed,
        "concurrency": concurrency,
        "retry_limit": 0,
        "noise": NoiseConfig(
            action_drop=0.0,
            observation_loss=0.0,
            burst_probability=0.0,
            burst_length=config.noise.burst_length,
        ),
    }
    for name, value in {
        "max_turns": max_turns,
        "max_tool_calls": max_tool_calls,
        "max_context_tokens": max_context_tokens,
        "max_generated_tokens": max_generated_tokens,
        "max_tokens_per_turn": max_tokens_per_turn,
    }.items():
        if value is not None:
            _require_positive(name, value)
            overrides[name] = value
    return replace(config, **overrides)


def _task_metadata(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "scenario": task["scenario"],
        "task_idx": task["task_idx"],
        "split": task.get("split"),
    }


def _status_for_trajectory(trajectory: Trajectory) -> str:
    for step in trajectory.steps:
        info = step.get("environment_info", {})
        awm = info.get("awm", {}) if isinstance(info, dict) else {}
        if isinstance(awm, dict) and awm.get("tool_terminal_failure"):
            return "tool_terminal_failure"
    if trajectory.success:
        return "success"
    if any(bool(step.get("format_error")) for step in trajectory.steps):
        return "format_error"
    if trajectory.tool_calls == 0:
        return "no_business_tool_call"
    return "completed_non_success"


def _exception_row(
    task: dict[str, Any],
    plan: Any,
    attempt: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "phase": "awm_model_replay",
        "task": _task_metadata(task),
        "attempt": attempt,
        "plan": plan.to_dict(),
        "status": "episode_exception",
        "success": False,
        "termination": "exception",
        "error": f"{type(error).__name__}: {error}",
        "steps": [],
    }


def _trajectory_row(
    trajectory: Trajectory,
    task: dict[str, Any],
    plan: Any,
    config: ExperimentConfig,
    attempt: int,
    checkpoint: str,
) -> dict[str, Any]:
    row = episode_record(trajectory, plan, config, checkpoint=checkpoint)
    row.update(
        schema=6,
        phase="awm_model_replay",
        task=_task_metadata(task),
        attempt=attempt,
        status=_status_for_trajectory(trajectory),
        # Retain the unescaped observation.  The SFT builder applies the same
        # Qwen control-token escaping as run_episode before serializing it.
        initial_observation=trajectory.initial_observation,
        initial_prompt_tokens=len(trajectory.prompt_tokens),
        initial_prompt_sha256=hashlib.sha256(
            trajectory.prompt_text.encode("utf-8")
        ).hexdigest(),
    )
    return row


def _terminal_failure_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract exact business calls that received terminal AWM failures."""
    task = row.get("task", {})
    if not isinstance(task, dict):
        return []
    candidates = []
    for step in row.get("steps", []):
        if not isinstance(step, dict):
            continue
        info = step.get("environment_info", {})
        awm = info.get("awm", {}) if isinstance(info, dict) else {}
        if not isinstance(awm, dict) or not awm.get("tool_terminal_failure"):
            continue
        try:
            action = json.loads(step.get("action", ""))
        except (TypeError, ValueError):
            continue
        if not isinstance(action, dict):
            continue
        tool_name, arguments = action.get("tool_name"), action.get("arguments")
        if not isinstance(tool_name, str) or not tool_name or not isinstance(arguments, dict):
            continue
        if not isinstance(task.get("scenario"), str) or type(task.get("task_idx")) is not int:
            continue
        candidates.append(
            {
                "scenario": task["scenario"],
                "task_idx": task["task_idx"],
                "task_id": task.get("id"),
                "tool_name": tool_name,
                "arguments": arguments,
                "failure_type": str(awm.get("tool_terminal_failure_type", "unknown")),
                "failure_error": str(awm.get("tool_terminal_failure_error", "")),
                "attempt": row.get("attempt"),
            }
        )
    return candidates


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        candidate["scenario"],
        candidate["task_idx"],
        candidate["tool_name"],
        json.dumps(candidate["arguments"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _accumulate_candidate(
    candidates: dict[tuple[str, int, str, str], dict[str, Any]],
    candidate: dict[str, Any],
    *,
    replay_output: Path,
) -> None:
    key = _candidate_key(candidate)
    aggregate = candidates.get(key)
    if aggregate is None:
        aggregate = {
            "kind": "task_tool_incident",
            "scenario": candidate["scenario"],
            "task_idx": candidate["task_idx"],
            "tool_name": candidate["tool_name"],
            "arguments": candidate["arguments"],
            "required_for_task": False,
            "excluded_from_training": False,
            "source_kind": "model_rollout_candidate",
            "task_id": candidate.get("task_id"),
            "evidence_output": str(replay_output),
            "observations": 0,
            "attempts": [],
            "failure_types": [],
            "error_examples": [],
            "review_note": (
                "Candidate only: inspect the saved model trace and establish whether this exact "
                "action is required before setting required_for_task=true."
            ),
        }
        candidates[key] = aggregate
    aggregate["observations"] += 1
    if candidate["attempt"] not in aggregate["attempts"]:
        aggregate["attempts"].append(candidate["attempt"])
    if candidate["failure_type"] not in aggregate["failure_types"]:
        aggregate["failure_types"].append(candidate["failure_type"])
    error = candidate["failure_error"]
    if error and error not in aggregate["error_examples"] and len(aggregate["error_examples"]) < 3:
        aggregate["error_examples"].append(error)


def _new_output_paths(output: Path, incident_output: Path | None) -> tuple[Path, Path, Path]:
    output = output.expanduser()
    summary = Path(str(output) + ".summary.json")
    default_incidents = output.with_suffix("")
    incidents = incident_output.expanduser() if incident_output else Path(str(default_incidents) + ".incidents.jsonl")
    paths = (output, summary, incidents)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Replay output, summary, and incident-output paths must be distinct")
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing replay artifact: {existing[0]}")
    return paths


async def _replay_attempts(
    attempts: list[tuple[int, dict[str, Any], int]],
    *,
    config: ExperimentConfig,
    tokenizer: Any,
    client: Any,
    checkpoint: str,
    temperature: float,
    output: Path,
    incident_output: Path,
    summary_output: Path,
    episode_runner: Callable[..., Any],
) -> dict[str, Any]:
    """Run bounded batches and stream durable rows after every completed batch."""
    candidates: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    exception_counts: Counter[str] = Counter()
    failure_type_counts: Counter[str] = Counter()
    successful = completed = generated_tokens = tool_calls = 0
    elapsed: list[float] = []

    async def one(index: int, record: dict[str, Any], attempt: int) -> dict[str, Any]:
        task = record["metadata"]["task"]
        plan = plan_sample(config, task["id"], index, attempt, evaluation=True)
        try:
            trajectory = await episode_runner(
                task,
                plan,
                config,
                tokenizer,
                client,
                {"temperature": temperature, "top_p": 1.0, "top_k": -1},
            )
            return _trajectory_row(trajectory, task, plan, config, attempt, checkpoint)
        except BaseException as exc:
            return _exception_row(task, plan, attempt, exc)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        batch_size = config.concurrency
        for start in range(0, len(attempts), batch_size):
            batch = attempts[start : start + batch_size]
            rows = await asyncio.gather(*(one(*item) for item in batch))
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                completed += 1
                status = str(row["status"])
                status_counts[status] += 1
                termination_counts[str(row.get("termination", "unknown"))] += 1
                if status == "episode_exception":
                    error = str(row.get("error", "unknown"))
                    exception_counts[error.split(":", 1)[0]] += 1
                else:
                    successful += int(bool(row.get("success")))
                    generated_tokens += int(row.get("generated_tokens", 0))
                    tool_calls += int(row.get("tool_calls", 0))
                    elapsed.append(float(row.get("elapsed_seconds", 0.0)))
                for candidate in _terminal_failure_candidates(row):
                    failure_type_counts[candidate["failure_type"]] += 1
                    _accumulate_candidate(candidates, candidate, replay_output=output)
            stream.flush()
            os.fsync(stream.fileno())
            print(
                f"AWM model replay: {completed}/{len(attempts)} attempts; "
                f"terminal-tool-failures={sum(failure_type_counts.values())}; "
                f"candidates={len(candidates)}",
                file=sys.stderr,
                flush=True,
            )

    incident_output.parent.mkdir(parents=True, exist_ok=True)
    with incident_output.open("x", encoding="utf-8") as stream:
        for _key, candidate in sorted(candidates.items()):
            candidate["attempts"].sort()
            candidate["failure_types"].sort()
            stream.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    tasks = {record["metadata"]["task"]["id"] for _, record, _ in attempts}
    summary = {
        "schema": 1,
        "phase": "awm_model_replay",
        "output": str(output.resolve()),
        "candidate_incidents": str(incident_output.resolve()),
        "tasks": len(tasks),
        "attempts_requested": len(attempts),
        "attempts_completed": completed,
        "successes": successful,
        "success_rate": successful / completed if completed else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "termination_counts": dict(sorted(termination_counts.items())),
        "episode_exception_types": dict(sorted(exception_counts.items())),
        "terminal_tool_failure_counts": dict(sorted(failure_type_counts.items())),
        "candidate_incident_count": len(candidates),
        "candidate_task_count": len({(row["scenario"], row["task_idx"]) for row in candidates.values()}),
        "mean_generated_tokens": generated_tokens / completed if completed else 0.0,
        "mean_tool_calls": tool_calls / completed if completed else 0.0,
        "mean_elapsed_seconds": mean(elapsed) if elapsed else 0.0,
        "noise": config.noise.__dict__,
        "policy": {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": -1,
            "max_turns": config.max_turns,
            "max_tool_calls": config.max_tool_calls,
            "max_context_tokens": config.max_context_tokens,
            "max_generated_tokens": config.max_generated_tokens,
            "max_tokens_per_turn": config.max_tokens_per_turn,
        },
        "candidate_policy": (
            "All candidates are non-excluding by default. Review a full trace, establish that the "
            "specific action is required for the task, then copy it to a reviewed incident file with "
            "required_for_task=true and run scripts/preflight_awm_tasks.sh."
        ),
    }
    atomic_json(summary_output, summary)
    return summary


def run_model_replay(
    data: str | Path,
    output: str | Path,
    *,
    config: ExperimentConfig | None = None,
    config_path: str | Path | None = None,
    model: str | Path | None = None,
    url: str,
    awm_url: str,
    repeats: int = 1,
    seed: int = 20260918,
    temperature: float = 0.2,
    concurrency: int = 8,
    shard_count: int = 1,
    shard_index: int = 0,
    tasks: int = 0,
    max_turns: int | None = None,
    max_tool_calls: int | None = None,
    max_context_tokens: int | None = None,
    max_generated_tokens: int | None = None,
    max_tokens_per_turn: int | None = None,
    incident_output: str | Path | None = None,
    tokenizer: Any | None = None,
    client_factory: Callable[..., Any] = SGLangClient,
    episode_runner: Callable[..., Any] = run_episode,
) -> dict[str, Any]:
    """Perform a full, fixed-policy AWM replay and create new evidence files."""
    _require_positive("repeats", repeats)
    _require_positive("concurrency", concurrency)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature < 0:
        raise ValueError("temperature must be nonnegative")
    data_path = Path(data).expanduser().resolve(strict=True)
    output_path, summary_path, incidents_path = _new_output_paths(Path(output), Path(incident_output) if incident_output else None)
    records = read_records(data_path)
    if any(record["metadata"]["task"]["environment"] != "awm" for record in records):
        raise ValueError("AWM model replay accepts an AWM-only manifest")
    selected = _selected_records(records, shard_count=shard_count, shard_index=shard_index, tasks=tasks)
    if config is None:
        if config_path is None:
            raise ValueError("config_path is required when config is not supplied")
        config = load_config(config_path)
    effective_config = _replay_config(
        config,
        url=awm_url,
        seed=seed,
        concurrency=concurrency,
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        max_context_tokens=max_context_tokens,
        max_generated_tokens=max_generated_tokens,
        max_tokens_per_turn=max_tokens_per_turn,
    )
    tokenizer = tokenizer or load_local_tokenizer(model)
    checkpoint = str(Path(model).expanduser().resolve()) if model else "injected-tokenizer"
    attempts = [
        (index, record, attempt)
        for index, record in enumerate(selected)
        for attempt in range(repeats)
    ]

    async def execute() -> dict[str, Any]:
        client = client_factory(url, effective_config.request_timeout)
        try:
            return await _replay_attempts(
                attempts,
                config=effective_config,
                tokenizer=tokenizer,
                client=client,
                checkpoint=checkpoint,
                temperature=float(temperature),
                output=output_path,
                incident_output=incidents_path,
                summary_output=summary_path,
                episode_runner=episode_runner,
            )
        finally:
            await client.close()

    summary = asyncio.run(execute())
    summary.update(
        data=str(data_path),
        data_sha256=_sha256(data_path),
        model=checkpoint,
        sglang_url=url,
        awm_url=awm_url,
        repeats=repeats,
        seed=seed,
        concurrency=concurrency,
        shard={"count": shard_count, "index": shard_index, "tasks_selected": len(selected)},
    )
    # The first write happens inside the coroutine so that an interrupted pass
    # still retains its stream.  Replace it with the complete provenance only
    # after the client has closed successfully.
    atomic_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="local AWM-only manifest")
    parser.add_argument("--output", required=True, type=Path, help="new JSONL file for complete model traces")
    parser.add_argument("--config", required=True, type=Path, help="existing AWM rollout YAML config")
    parser.add_argument("--model", required=True, type=Path, help="local Qwen tokenizer directory matching SGLang")
    parser.add_argument("--url", required=True, help="already-running SGLang server URL")
    parser.add_argument("--awm-url", required=True, help="already-running local AWM server URL")
    parser.add_argument("--incident-output", type=Path, help="new JSONL candidate incident output")
    parser.add_argument("--repeats", type=int, default=1, help="independent fresh-session attempts per task")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--concurrency", type=int, default=8, help="maximum simultaneous AWM/model trajectories")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=0, help="0 means every task in the selected shard")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--max-generated-tokens", type=int)
    parser.add_argument("--max-tokens-per-turn", type=int)
    args = parser.parse_args(argv)
    result = run_model_replay(
        args.data,
        args.output,
        config_path=args.config,
        model=args.model,
        url=args.url,
        awm_url=args.awm_url,
        incident_output=args.incident_output,
        repeats=args.repeats,
        seed=args.seed,
        temperature=args.temperature,
        concurrency=args.concurrency,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        tasks=args.tasks,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        max_context_tokens=args.max_context_tokens,
        max_generated_tokens=args.max_generated_tokens,
        max_tokens_per_turn=args.max_tokens_per_turn,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
