"""Build verified, local-only ALFWorld planner demonstrations for Slime SFT.

The official ALFWorld game generator saves a planner ``walkthrough`` inside
each solvable ``game.tw-pddl``.  This module treats that immutable field as the
primary expert source, then replays it through the exact TextWorld wrapper used
by online RL.  A row is emitted only if the replay reaches a verified terminal
success; this avoids teaching the model actions that merely look plausible in
PDDL but do not match the text-game harness.
"""

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from .agent import SYSTEM_PROMPT
from .data import alfworld_records, atomic_json, read_records, validate_local_records
from .envs import ALFWorldEnvironment, TextEnvironment, canonical_action, parse_action

SFT_SCHEMA_VERSION = 1
_ALLOWED_ROLES = {"system", "user", "assistant"}


def action_json(action: str) -> str:
    """Serialize an action exactly as the online harness asks the policy to emit it."""
    action = canonical_action(action)
    if not action:
        raise ValueError("Expert action must be nonempty")
    value = json.dumps({"action": action}, ensure_ascii=False, separators=(",", ":"))
    if parse_action(value) != action:
        raise ValueError(f"Expert action cannot be represented by the policy protocol: {action!r}")
    return value


def walkthrough_from_game(task: dict) -> list[str] | None:
    """Read the stored official planner solution without changing the game file."""
    game = Path(task["gamefile"]).expanduser().resolve(strict=True)
    content = json.loads(game.read_text(encoding="utf-8"))
    walkthrough = content.get("walkthrough")
    if content.get("solvable") is not True or not isinstance(walkthrough, list) or not walkthrough:
        return None
    if not all(isinstance(action, str) for action in walkthrough):
        raise ValueError(f"ALFWorld walkthrough has a non-string action: {game}")
    return [canonical_action(action) for action in walkthrough]


def planner_walkthrough(task: dict) -> list[str]:
    """Ask ALFWorld's local PDDL planner for a plan without writing dataset files.

    This is intentionally an opt-in fallback.  Normal data construction reads
    the existing ``walkthrough`` so repeated training runs do not spend CPU on
    FastDownward or alter the local ALFWorld dataset.
    """
    import textworld
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
        AlfredInfos,
    )
    from textworld.envs import PddlEnv

    game = Path(task["gamefile"]).expanduser().resolve(strict=True)
    env = PddlEnv(textworld.EnvInfos(admissible_commands=True, extras=["gamefile"]))
    env = AlfredDemangler(env, shuffle=False)
    env = AlfredInfos(env)
    env = AlfredExpert(env, expert_type=AlfredExpertType.PLANNER)
    try:
        env.load(str(game))
        state = env.reset()
        plan = state["extra.expert_plan"]
    finally:
        env.close()
    if not isinstance(plan, list) or not plan or not all(isinstance(action, str) for action in plan):
        raise ValueError(f"Local ALFWorld planner returned no usable plan: {game}")
    return [canonical_action(action) for action in plan]


def build_messages(
    task: dict,
    walkthrough: Iterable[str],
    environment_factory: Callable[[dict], TextEnvironment] = ALFWorldEnvironment,
) -> tuple[list[dict], int]:
    """Replay one expert plan and create a multi-turn, loss-masked SFT row."""
    actions = [canonical_action(action) for action in walkthrough]
    if not actions:
        raise ValueError("Expert plan is empty")
    messages = [{"role": "system", "content": SYSTEM_PROMPT, "step_loss_mask": 0}]
    environment = environment_factory(task)
    try:
        initial = environment.reset()
        if not initial.observation.strip():
            raise ValueError("ALFWorld reset returned an empty observation")
        if initial.success:
            raise ValueError("ALFWorld expert game is already solved at reset")
        messages.append({"role": "user", "content": initial.observation, "step_loss_mask": 0})
        for position, action in enumerate(actions, 1):
            messages.append({"role": "assistant", "content": action_json(action), "step_loss_mask": 1})
            result = environment.step(action)
            if result.success or result.terminated:
                if not result.success:
                    raise ValueError(f"Expert plan terminated without success at action {position}")
                if position != len(actions):
                    raise ValueError(f"Expert plan succeeded before its final action ({position}/{len(actions)})")
                return messages, len(actions)
            if not result.observation.strip():
                raise ValueError(f"ALFWorld returned an empty observation after action {position}")
            messages.append({"role": "user", "content": result.observation, "step_loss_mask": 0})
    finally:
        environment.close()
    raise ValueError(f"Expert plan did not solve the task after {len(actions)} actions")


def sft_row(task: dict, walkthrough: Iterable[str], source: str, environment_factory=ALFWorldEnvironment) -> dict:
    messages, action_count = build_messages(task, walkthrough, environment_factory)
    return {
        "messages": messages,
        "metadata": {
            "schema": SFT_SCHEMA_VERSION,
            "dataset": "alfworld_planner_sft",
            "source": source,
            "task_id": task["id"],
            "split": task["split"],
            "task_type": task.get("task_type"),
            "gamefile": task["gamefile"],
            "game_sha256": task.get("game_sha256"),
            "action_count": action_count,
        },
    }


def validate_sft_records(path: str | Path) -> dict:
    """Validate the native Slime multi-turn SFT schema and return dataset facts."""
    path = Path(path).expanduser().resolve(strict=True)
    rows = 0
    actions = 0
    task_ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                messages = row["messages"]
                metadata = row["metadata"]
                if not isinstance(messages, list) or len(messages) < 3:
                    raise ValueError("messages must include system, user, and at least one assistant action")
                if (
                    messages[0].get("role") != "system"
                    or messages[0].get("step_loss_mask") != 0
                    or not isinstance(messages[0].get("content"), str)
                    or not messages[0]["content"].strip()
                ):
                    raise ValueError("first system message must have step_loss_mask=0")
                if (
                    messages[1].get("role") != "user"
                    or messages[1].get("step_loss_mask") != 0
                    or not isinstance(messages[1].get("content"), str)
                    or not messages[1]["content"].strip()
                ):
                    raise ValueError("system must be followed by a masked user observation")
                if messages[-1].get("role") != "assistant":
                    raise ValueError("trajectory must end in the final expert action")
                expected_role = "assistant"
                for message in messages[2:]:
                    role = message.get("role")
                    if role not in _ALLOWED_ROLES or role != expected_role:
                        raise ValueError("messages after reset must alternate assistant action and user observation")
                    if not isinstance(message.get("content"), str) or not message["content"].strip():
                        raise ValueError("message content must be nonempty text")
                    expected_mask = 1 if role == "assistant" else 0
                    if message.get("step_loss_mask") != expected_mask:
                        raise ValueError(f"{role} messages must have step_loss_mask={expected_mask}")
                    if role == "assistant":
                        action = parse_action(message["content"])
                        if message["content"] != action_json(action):
                            raise ValueError("assistant action is not canonical JSON")
                        actions += 1
                    expected_role = "user" if expected_role == "assistant" else "assistant"
                if not isinstance(metadata, dict) or metadata.get("schema") != SFT_SCHEMA_VERSION:
                    raise ValueError("missing or unsupported metadata schema")
                if metadata.get("dataset") != "alfworld_planner_sft":
                    raise ValueError("unexpected SFT dataset name")
                task_id = metadata.get("task_id")
                if not isinstance(task_id, str) or not task_id or task_id in task_ids:
                    raise ValueError("metadata.task_id must be nonempty and unique")
                if metadata.get("split") != "train":
                    raise ValueError("SFT training records must come only from the ALFWorld train split")
                task_ids.add(task_id)
                rows += 1
            except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid SFT data {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"SFT dataset is empty: {path}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "records": rows,
        "expert_actions": actions,
    }


def _write_rows(path: Path, rows: Iterable[dict]) -> None:
    """Atomically create a new JSONL dataset without ever replacing prior data."""
    if path.exists():
        raise FileExistsError(f"SFT output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Recheck avoids silently replacing an output created while building.
        if path.exists():
            raise FileExistsError(f"SFT output already exists: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_sft_dataset(
    records: list[dict],
    output: str | Path,
    *,
    planner_fallback: bool = False,
    max_actions: int = 40,
    strict: bool = False,
    environment_factory: Callable[[dict], TextEnvironment] = ALFWorldEnvironment,
) -> dict:
    """Build and verify rows. Invalid or stale local games are reported and excluded."""
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    if not records:
        raise ValueError("No records were supplied")
    validate_local_records(records)
    if any(record["metadata"]["task"].get("split") != "train" for record in records):
        raise ValueError("Build SFT only from the ALFWorld train split")

    output = Path(output).expanduser().resolve()
    report_path = Path(str(output) + ".report.json")
    if report_path.exists():
        raise FileExistsError(f"SFT report already exists: {report_path}")
    rows = []
    source_counts = Counter()
    skipped = Counter()
    skipped_examples = []
    for record in records:
        task = record["metadata"]["task"]
        try:
            walkthrough = walkthrough_from_game(task)
            source = "game.tw-pddl.walkthrough"
            if walkthrough is None:
                if not planner_fallback:
                    raise ValueError("missing stored planner walkthrough")
                walkthrough = planner_walkthrough(task)
                source = "local_pddl_planner_fallback"
            if len(walkthrough) > max_actions:
                raise ValueError(f"expert plan has {len(walkthrough)} actions; max_actions={max_actions}")
            row = sft_row(task, walkthrough, source, environment_factory)
            rows.append(row)
            source_counts[source] += 1
        except Exception as exc:  # Continue only with independently replay-verified examples.
            if strict:
                raise RuntimeError(f"Could not build expert SFT row for {task['id']}: {exc}") from exc
            reason = type(exc).__name__
            skipped[reason] += 1
            if len(skipped_examples) < 20:
                skipped_examples.append({"task_id": task["id"], "reason": str(exc)})
    if not rows:
        raise ValueError("No verified expert trajectories were produced; inspect ALFWorld game files and builder settings")
    _write_rows(output, rows)
    facts = validate_sft_records(output)
    report = {
        "schema": SFT_SCHEMA_VERSION,
        "source_manifest_records": len(records),
        "verified_records": facts["records"],
        "expert_actions": facts["expert_actions"],
        "dataset_sha256": facts["sha256"],
        "sources": dict(source_counts),
        "skipped": dict(skipped),
        "skipped_examples": skipped_examples,
        "planner_fallback": planner_fallback,
        "max_actions": max_actions,
        "output": facts["path"],
    }
    atomic_json(report_path, report)
    return report


def _records_from_args(args) -> list[dict]:
    if args.manifest:
        records = read_records(args.manifest)
    else:
        records = alfworld_records(args.root, "train", args.limit)
    if args.limit is not None and args.manifest:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        records = records[: args.limit]
    return records


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Existing local ALFWorld train manifest")
    source.add_argument("--root", help="Local ALFWorld json_2.1.1 directory")
    parser.add_argument("--limit", type=int, help="Optional deterministic subset size when --root is used")
    parser.add_argument("--output", required=True, help="New Slime SFT JSONL path")
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument(
        "--planner-fallback",
        action="store_true",
        help="Use the local PDDL planner only when a game lacks its stored walkthrough",
    )
    parser.add_argument("--strict", action="store_true", help="Stop at the first invalid expert game")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    report = build_sft_dataset(
        _records_from_args(args),
        args.output,
        planner_fallback=args.planner_fallback,
        max_actions=args.max_actions,
        strict=args.strict,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
