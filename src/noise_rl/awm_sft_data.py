"""Build audited, action-only Slime SFT data from successful AWM model replays.

The source is a completed ``noise_rl.awm_model_replay`` JSONL, not an AWM
task manifest.  A row is eligible only when its *code* verifier reached a
verified terminal success in a clean, fault-free session.  The emitted chat
messages reproduce the production AWM harness exactly: the AWM system prompt,
the reset/tool-discovery observation, canonical tool JSON, and the subsequent
observations.  Only assistant tool actions have ``step_loss_mask=1``.

New replay rows persist ``initial_observation``.  Older rows can be rebuilt
against a running local AWM service, but their recorded prompt SHA-256 and
token count must still match the regenerated prompt.  This prevents silently
turning a changed tool schema into a superficially plausible SFT example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from .agent import QwenChatProtocol
from .awm import AWMEnvironment, SYSTEM_PROMPT, canonical_action
from .awm_context import load_local_tokenizer

AWM_SFT_SCHEMA_VERSION = 1
AWM_SFT_DATASET = "awm_verified_replay_sft"
_ALLOWED_ROLES = {"system", "user", "assistant"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_nonempty_text(value: object, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    return value


def _safe_observation(value: str) -> str:
    """Use precisely the same control-token escaping as ``run_episode``."""
    return QwenChatProtocol.safe_observation(value)


def _task_from_row(row: dict[str, Any]) -> dict[str, Any]:
    task = row.get("task")
    if not isinstance(task, dict):
        raise ValueError("missing task metadata")
    task_id = _require_nonempty_text(task.get("id"), "task.id must be nonempty")
    scenario = _require_nonempty_text(task.get("scenario"), "task.scenario must be nonempty")
    task_idx = task.get("task_idx")
    if type(task_idx) is not int or task_idx < 0:
        raise ValueError("task.task_idx must be a nonnegative integer")
    if task.get("split") != "train":
        raise ValueError("only train-split replay trajectories are allowed")
    return {"id": task_id, "scenario": scenario, "task_idx": task_idx, "split": "train"}


def _is_clean_noise(row: dict[str, Any]) -> bool:
    config = row.get("config")
    noise = config.get("noise") if isinstance(config, dict) else None
    if not isinstance(noise, dict):
        return False
    return all(float(noise.get(name, -1)) == 0.0 for name in ("action_drop", "observation_loss", "burst_probability"))


def _has_injected_fault(row: dict[str, Any]) -> bool:
    audit = row.get("fault_audit", [])
    if not isinstance(audit, list):
        return True
    for item in audit:
        if not isinstance(item, dict):
            return True
        if item.get("dropped") or item.get("action_dropped") or item.get("observation_loss_draw") or item.get("observation_lost"):
            return True
    return False


def _validate_source_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("phase") != "awm_model_replay" or row.get("environment") != "awm":
        raise ValueError("not an AWM model replay row")
    if row.get("status") != "success" or row.get("success") is not True or row.get("termination") != "success":
        raise ValueError("trajectory is not a completed success")
    if not _is_clean_noise(row) or _has_injected_fault(row):
        raise ValueError("trajectory was not collected with clean fault-free AWM settings")
    return _task_from_row(row)


def _step_actions(row: dict[str, Any], *, max_actions: int) -> list[tuple[str, str]]:
    steps = row.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("successful trajectory has no steps")
    if row.get("format_errors") != 0:
        raise ValueError("trajectory contains one or more format errors")
    if len(steps) > max_actions:
        raise ValueError(f"trajectory has more than max_actions={max_actions} actions")
    parsed: list[tuple[str, str]] = []
    for position, step in enumerate(steps, 1):
        if not isinstance(step, dict) or step.get("format_error") is not False:
            raise ValueError(f"step {position} is not a valid harness action")
        action = _require_nonempty_text(step.get("action"), f"step {position} action is empty")
        canonical = canonical_action(action)
        if action != canonical:
            raise ValueError(f"step {position} action is not canonical online-harness JSON")
        observation = _require_nonempty_text(step.get("observation"), f"step {position} observation is empty")
        environment_info = step.get("environment_info")
        awm_info = environment_info.get("awm") if isinstance(environment_info, dict) else None
        if isinstance(awm_info, dict) and awm_info.get("tool_terminal_failure"):
            raise ValueError(f"step {position} has a terminal AWM tool failure")
        try:
            observation_value = json.loads(observation)
        except (TypeError, ValueError):
            observation_value = None
        if isinstance(observation_value, dict) and observation_value.get("error"):
            raise ValueError(f"step {position} returned a business-tool error")
        if re.search(r"(?i)\b(?:http\s*)?(?:422|500)\b|\btimeout\b|timed out", observation):
            raise ValueError(f"step {position} observation contains 422/500/timeout evidence")
        parsed.append((canonical, observation))

    done = json.loads(parsed[-1][0])
    if done["tool_name"] != "done":
        raise ValueError("successful trajectory does not end with done")
    if any(json.loads(action)["tool_name"] == "done" for action, _ in parsed[:-1]):
        raise ValueError("done appears before the final action")
    final_info = steps[-1].get("environment_info")
    awm = final_info.get("awm") if isinstance(final_info, dict) else None
    if not isinstance(awm, dict) or awm.get("verifier_reward_type") != "complete":
        raise ValueError("terminal success lacks a completed AWM code-verifier result")
    execution = awm.get("verify_execution_status")
    if execution is not None and execution != "success":
        raise ValueError("terminal code verifier did not execute successfully")
    return parsed


def _rebuild_initial_observation(task: dict[str, Any], url: str, timeout: float) -> str:
    environment = AWMEnvironment(task, url, timeout=timeout)
    try:
        result = environment.reset()
        return _require_nonempty_text(result.observation, "AWM reset returned an empty initial observation")
    finally:
        environment.close()


def _initial_observation(
    row: dict[str, Any],
    task: dict[str, Any],
    *,
    awm_url: str | None,
    rehydrate_timeout: float,
    resolver: Callable[[dict[str, Any]], str] | None,
) -> tuple[str, str]:
    stored = row.get("initial_observation")
    if isinstance(stored, str) and stored.strip():
        return stored, "stored"
    if resolver is not None:
        return _require_nonempty_text(resolver(task), "initial-observation resolver returned empty text"), "rehydrated"
    if not awm_url:
        raise ValueError(
            "legacy replay lacks initial_observation; pass --awm-url to rehydrate against the local AWM server"
        )
    return _rebuild_initial_observation(task, awm_url, rehydrate_timeout), "rehydrated"


def make_prompt_auditor(tokenizer) -> Callable[[dict[str, Any], str], None]:
    """Require regenerated source prompt bytes and token count to match replay."""
    protocol = QwenChatProtocol(tokenizer)

    def audit(row: dict[str, Any], observation: str) -> None:
        expected_hash = row.get("initial_prompt_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("source row has no valid initial_prompt_sha256")
        prompt, tokens = protocol.initial(observation, SYSTEM_PROMPT)
        actual_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("regenerated initial prompt differs from the recorded online-harness prompt")
        expected_tokens = row.get("initial_prompt_tokens")
        if type(expected_tokens) is not int or expected_tokens < 1:
            raise ValueError("source row has no valid initial_prompt_tokens")
        if len(tokens) != expected_tokens:
            raise ValueError("regenerated initial prompt token count differs from replay")

    return audit


def _sft_row(
    row: dict[str, Any],
    task: dict[str, Any],
    *,
    source_line: int,
    replay_path: Path,
    replay_sha256: str,
    initial_observation: str,
    actions: list[tuple[str, str]],
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT, "step_loss_mask": 0},
        {"role": "user", "content": _safe_observation(initial_observation), "step_loss_mask": 0},
    ]
    for position, (action, observation) in enumerate(actions):
        messages.append({"role": "assistant", "content": action, "step_loss_mask": 1})
        if position + 1 < len(actions):
            messages.append({"role": "user", "content": _safe_observation(observation), "step_loss_mask": 0})
    return {
        "messages": messages,
        "metadata": {
            "schema": AWM_SFT_SCHEMA_VERSION,
            "dataset": AWM_SFT_DATASET,
            "task_id": task["id"],
            "environment": "awm",
            "split": "train",
            "scenario": task["scenario"],
            "task_idx": task["task_idx"],
            "action_count": len(actions),
            "verifier": {"mode": "code", "reward_type": "complete"},
            "source": {
                "replay_path": str(replay_path),
                "replay_sha256": replay_sha256,
                "replay_line": source_line,
                "attempt": row.get("attempt"),
                "checkpoint": row.get("checkpoint"),
                "initial_prompt_sha256": row.get("initial_prompt_sha256"),
                "initial_prompt_tokens": row.get("initial_prompt_tokens"),
            },
        },
    }


def validate_awm_sft_records(path: str | Path) -> dict[str, Any]:
    """Validate the AWM replay SFT schema before Slime is allowed to train."""
    path = Path(path).expanduser().resolve(strict=True)
    rows = actions = 0
    task_ids: set[str] = set()
    scenarios: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                messages, metadata = row["messages"], row["metadata"]
                if not isinstance(messages, list) or len(messages) < 3:
                    raise ValueError("messages must include system, user, and a final done action")
                if messages[0] != {"role": "system", "content": SYSTEM_PROMPT, "step_loss_mask": 0}:
                    raise ValueError("system message differs from the online AWM harness")
                if messages[1].get("role") != "user" or messages[1].get("step_loss_mask") != 0:
                    raise ValueError("system must be followed by a masked reset observation")
                _require_nonempty_text(messages[1].get("content"), "reset observation is empty")
                expected_role = "assistant"
                action_count = 0
                for message in messages[2:]:
                    if not isinstance(message, dict) or message.get("role") not in _ALLOWED_ROLES:
                        raise ValueError("invalid message")
                    if message.get("role") != expected_role:
                        raise ValueError("messages must alternate assistant action and user observation")
                    expected_mask = 1 if expected_role == "assistant" else 0
                    if message.get("step_loss_mask") != expected_mask:
                        raise ValueError("message loss mask does not match its role")
                    content = _require_nonempty_text(message.get("content"), "message content is empty")
                    if expected_role == "assistant":
                        if content != canonical_action(content):
                            raise ValueError("assistant action is not canonical online-harness JSON")
                        action_count += 1
                    expected_role = "user" if expected_role == "assistant" else "assistant"
                if messages[-1].get("role") != "assistant" or json.loads(messages[-1]["content"])["tool_name"] != "done":
                    raise ValueError("SFT trajectory must end with a done action")
                if not isinstance(metadata, dict) or metadata.get("schema") != AWM_SFT_SCHEMA_VERSION:
                    raise ValueError("missing or unsupported metadata schema")
                if metadata.get("dataset") != AWM_SFT_DATASET or metadata.get("environment") != "awm":
                    raise ValueError("unexpected SFT dataset or environment")
                if metadata.get("split") != "train":
                    raise ValueError("SFT data must use the AWM train split")
                task_id = _require_nonempty_text(metadata.get("task_id"), "metadata.task_id must be nonempty")
                if task_id in task_ids:
                    raise ValueError("metadata.task_id must be unique")
                if type(metadata.get("task_idx")) is not int or not isinstance(metadata.get("scenario"), str):
                    raise ValueError("missing AWM task identity")
                if metadata.get("action_count") != action_count:
                    raise ValueError("metadata.action_count does not match messages")
                verifier = metadata.get("verifier")
                source = metadata.get("source")
                if verifier != {"mode": "code", "reward_type": "complete"} or not isinstance(source, dict):
                    raise ValueError("missing code-verifier or replay provenance")
                if not isinstance(source.get("replay_path"), str) or not source["replay_path"]:
                    raise ValueError("missing source.replay_path")
                if not isinstance(source.get("replay_sha256"), str) or len(source["replay_sha256"]) != 64:
                    raise ValueError("missing source.replay_sha256")
                if type(source.get("replay_line")) is not int or source["replay_line"] < 1:
                    raise ValueError("missing source.replay_line")
                if not isinstance(source.get("initial_prompt_sha256"), str) or len(source["initial_prompt_sha256"]) != 64:
                    raise ValueError("missing source.initial_prompt_sha256")
                if type(source.get("initial_prompt_tokens")) is not int or source["initial_prompt_tokens"] < 1:
                    raise ValueError("missing source.initial_prompt_tokens")
                task_ids.add(task_id)
                scenarios.add(metadata["scenario"])
                rows += 1
                actions += action_count
            except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid AWM SFT data {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"AWM SFT dataset is empty: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "dataset": AWM_SFT_DATASET,
        "records": rows,
        "expert_actions": actions,
        "unique_tasks": len(task_ids),
        "unique_scenarios": len(scenarios),
    }


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"AWM SFT output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"AWM SFT output already exists: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"AWM SFT report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build_awm_sft_dataset(
    replay: str | Path,
    output: str | Path,
    report: str | Path,
    *,
    max_actions: int = 40,
    max_per_task: int = 1,
    max_per_scenario: int = 20,
    awm_url: str | None = None,
    rehydrate_timeout: float = 180,
    prompt_audit: Callable[[dict[str, Any], str], None] | None = None,
    initial_observation_resolver: Callable[[dict[str, Any]], str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Create a non-overwriting, prompt-audited AWM SFT JSONL and report."""
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    if type(max_per_task) is not int or max_per_task < 1:
        raise ValueError("max_per_task must be a positive integer")
    if type(max_per_scenario) is not int or max_per_scenario < 1:
        raise ValueError("max_per_scenario must be a positive integer")
    if rehydrate_timeout <= 0:
        raise ValueError("rehydrate_timeout must be positive")
    if prompt_audit is None:
        raise ValueError("prompt_audit is required to verify the exact online-harness prompt")
    replay_path = Path(replay).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve()
    report_path = Path(report).expanduser().resolve()
    if output_path == report_path or output_path == replay_path or report_path == replay_path:
        raise ValueError("Replay, SFT output, and quality report must use distinct paths")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("AWM SFT output or report already exists; choose new paths")

    replay_sha256 = _sha256(replay_path)
    candidates: list[tuple[tuple[Any, ...], int, dict[str, Any], dict[str, Any], list[tuple[str, str]]]] = []
    rejected: Counter[str] = Counter()
    rejected_examples: list[dict[str, Any]] = []
    source_rows = 0
    with replay_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            source_rows += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not a JSON object")
                task = _validate_source_row(row)
                actions = _step_actions(row, max_actions=max_actions)
                # Prefer concise demonstrated solutions when a replay has repeats.
                rank = (len(actions), int(row.get("generated_tokens", 10**12)), task["id"], line_number)
                candidates.append((rank, line_number, row, task, actions))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                reason = str(exc) or type(exc).__name__
                rejected[reason] += 1
                if len(rejected_examples) < 30:
                    rejected_examples.append({"line": line_number, "reason": reason})
    selected: list[tuple[int, dict[str, Any], dict[str, Any], list[tuple[str, str]]]] = []
    per_task = Counter()
    per_scenario = Counter()
    for _rank, line_number, row, task, actions in sorted(candidates):
        if per_task[task["id"]] >= max_per_task:
            rejected["per_task_cap"] += 1
            continue
        if per_scenario[task["scenario"]] >= max_per_scenario:
            rejected["per_scenario_cap"] += 1
            continue
        per_task[task["id"]] += 1
        per_scenario[task["scenario"]] += 1
        selected.append((line_number, row, task, actions))

    rows: list[dict[str, Any]] = []
    initial_sources = Counter()
    for line_number, row, task, actions in selected:
        try:
            observation, source = _initial_observation(
                row,
                task,
                awm_url=awm_url,
                rehydrate_timeout=rehydrate_timeout,
                resolver=initial_observation_resolver,
            )
            prompt_audit(row, observation)
            initial_sources[source] += 1
            rows.append(
                _sft_row(
                    row,
                    task,
                    source_line=line_number,
                    replay_path=replay_path,
                    replay_sha256=replay_sha256,
                    initial_observation=observation,
                    actions=actions,
                )
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            reason = str(exc) or type(exc).__name__
            rejected[reason] += 1
            if len(rejected_examples) < 30:
                rejected_examples.append({"line": line_number, "task_id": task["id"], "reason": reason})
            if strict:
                raise ValueError(f"Rejected source replay line {line_number}: {reason}") from exc
    _write_rows(output_path, rows)
    if rows:
        facts = validate_awm_sft_records(output_path)
    else:
        facts = {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "dataset": AWM_SFT_DATASET,
            "records": 0,
            "expert_actions": 0,
            "unique_tasks": 0,
            "unique_scenarios": 0,
        }
    counts = [row["metadata"]["action_count"] for row in rows]
    scenario_counts = Counter(row["metadata"]["scenario"] for row in rows)
    business_tools = Counter()
    for row in rows:
        for message in row["messages"]:
            if message["role"] == "assistant":
                tool_name = json.loads(message["content"])["tool_name"]
                if tool_name != "done":
                    business_tools[tool_name] += 1
    report_value = {
        "schema": 1,
        "phase": "awm_sft_data_build",
        "dataset": AWM_SFT_DATASET,
        "replay": str(replay_path),
        "replay_sha256": replay_sha256,
        "source_rows": source_rows,
        "clean_code_verifier_success_candidates": len(candidates),
        "selected_after_caps": len(selected),
        "records": facts["records"],
        "unique_tasks": facts["unique_tasks"],
        "unique_scenarios": facts["unique_scenarios"],
        "expert_actions": facts["expert_actions"],
        "action_count": {
            "min": min(counts) if counts else None,
            "mean": mean(counts) if counts else None,
            "max": max(counts) if counts else None,
        },
        "initial_observation_source": dict(sorted(initial_sources.items())),
        "prompt_audit": "online_prompt_hash_and_token_count",
        "max_actions": max_actions,
        "max_per_task": max_per_task,
        "max_per_scenario": max_per_scenario,
        "records_per_scenario": dict(sorted(scenario_counts.items())),
        "scenario_record_count_distribution": {
            "min": min(scenario_counts.values()) if scenario_counts else None,
            "mean": mean(scenario_counts.values()) if scenario_counts else None,
            "max": max(scenario_counts.values()) if scenario_counts else None,
        },
        "business_tool_calls": dict(business_tools.most_common()),
        "rejected": dict(rejected.most_common()),
        "rejected_examples": rejected_examples,
        "output": str(output_path),
        "output_sha256": facts["sha256"],
    }
    _write_report(report_path, report_value)
    return report_value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, help="Completed AWM model replay JSONL")
    parser.add_argument("--output", required=True, help="New action-only Slime SFT JSONL")
    parser.add_argument("--report", required=True, help="New data-quality JSON report")
    parser.add_argument("--hf-checkpoint", required=True, help="Local Qwen tokenizer/checkpoint used in replay")
    parser.add_argument("--awm-url", help="Required only to rehydrate legacy replays without initial_observation")
    parser.add_argument("--rehydrate-timeout", type=float, default=180)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--max-per-task", type=int, default=1)
    parser.add_argument("--max-per-scenario", type=int, default=20)
    parser.add_argument("--strict", action="store_true", help="Fail on the first selected replay that cannot be reconstructed")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    tokenizer = load_local_tokenizer(args.hf_checkpoint)
    value = build_awm_sft_dataset(
        args.replay,
        args.output,
        args.report,
        max_actions=args.max_actions,
        max_per_task=args.max_per_task,
        max_per_scenario=args.max_per_scenario,
        awm_url=args.awm_url,
        rehydrate_timeout=args.rehydrate_timeout,
        prompt_audit=make_prompt_auditor(tokenizer),
        strict=args.strict,
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
