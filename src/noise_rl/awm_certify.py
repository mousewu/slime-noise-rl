"""Certify AWM tasks before they are admitted to SFT or RL training.

The pipeline separates five questions which a model-only replay cannot answer
on its own:

* structure: can a fresh task session reset repeatedly with one stable tool
  schema and a code verifier;
* witness: can an already verified teacher action trace succeed repeatedly in
  fresh sessions;
* verifier: does the verifier reject a no-op and at least one near-miss trace;
* policy stability: did a clean student replay avoid harness/environment
  failures; and
* learnability: is the student's clean success rate neither all-zero nor
  saturated.

Every live probe uses the same dependency-light :class:`AWMEnvironment` as
training.  The task-level evidence JSONL is append-only and resumable, while
the two frozen manifests and report are created only after every task has a
decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from .awm import AWMEnvironment, canonical_action
from .data import atomic_json, read_records, write_records


_ERROR_TEXT = re.compile(
    r"(?i)(?:environment_error|internal server error|status\s*code\s*[:=]?\s*(?:422|500)"
    r"|http(?:\s+status)?\s*[:=]?\s*(?:422|500)|\b(?:422|500)\s+(?:error|response)"
    r"|\btimeout\b|timed out)"
)
_INFRA_STATUSES = {"episode_exception", "tool_terminal_failure"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            value["_certification_source_line"] = line_number
            rows.append(value)
    if not rows:
        raise ValueError(f"Empty replay JSONL: {path}")
    return rows


def _task_key_from_manifest(record: dict[str, Any]) -> tuple[str, int]:
    task = record["metadata"]["task"]
    if task.get("environment") != "awm" or task.get("split") != "train":
        raise ValueError("AWM certification accepts only train-split AWM manifest records")
    return task["scenario"], task["task_idx"]


def _task_key_from_replay(row: dict[str, Any]) -> tuple[str, int]:
    task = row.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("scenario"), str):
        raise ValueError("Replay row lacks task.scenario")
    if type(task.get("task_idx")) is not int:
        raise ValueError("Replay row lacks integer task.task_idx")
    if task.get("split") != "train":
        raise ValueError("Certification replay must contain only train-split tasks")
    return task["scenario"], task["task_idx"]


def _noise_is_clean(row: dict[str, Any]) -> bool:
    config = row.get("config")
    noise = config.get("noise") if isinstance(config, dict) else None
    if not isinstance(noise, dict):
        return False
    return all(
        isinstance(noise.get(name), (int, float))
        and not isinstance(noise.get(name), bool)
        and float(noise[name]) == 0.0
        for name in ("action_drop", "observation_loss", "burst_probability")
    )


def _fault_audit_is_clean(row: dict[str, Any]) -> bool:
    audit = row.get("fault_audit", [])
    if not isinstance(audit, list):
        return False
    return not any(
        not isinstance(event, dict)
        or event.get("dropped")
        or event.get("action_dropped")
        or event.get("observation_lost")
        or event.get("observation_loss_draw")
        for event in audit
    )


def _observation_error(observation: Any) -> str | None:
    if not isinstance(observation, str):
        return "non_text_observation"
    if _ERROR_TEXT.search(observation):
        return "http_or_timeout_evidence"
    try:
        value = json.loads(observation)
    except (TypeError, ValueError):
        return None
    if isinstance(value, dict) and value.get("error") not in (None, "", False):
        return "business_tool_error"
    return None


def replay_row_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    """Return strict, nested replay diagnostics used by both evidence inputs."""
    environment_errors: list[str] = []
    format_errors: list[str] = []
    if row.get("status") in _INFRA_STATUSES:
        environment_errors.append(f"status:{row.get('status')}")
    if row.get("status") == "format_error":
        format_errors.append("status:format_error")
    if row.get("termination") == "exception":
        environment_errors.append("termination:exception")
    if int(row.get("format_errors", 0) or 0) > 0:
        format_errors.append("format_errors")
    if row.get("error"):
        error = str(row["error"])
        if _ERROR_TEXT.search(error) or row.get("status") == "episode_exception":
            environment_errors.append(f"exception:{error[:240]}")

    steps = row.get("steps", [])
    if not isinstance(steps, list):
        format_errors.append("steps:not_list")
        steps = []
    for position, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            format_errors.append(f"step:{position}:not_object")
            continue
        if step.get("format_error"):
            format_errors.append(f"step:{position}:format_error")
        info = step.get("environment_info")
        awm = info.get("awm") if isinstance(info, dict) else None
        if isinstance(awm, dict) and (
            awm.get("tool_terminal_failure")
            or awm.get("tool_terminal_failure_type")
        ):
            kind = awm.get("tool_terminal_failure_type", "unknown")
            environment_errors.append(f"step:{position}:tool_terminal_failure:{kind}")
        observation_error = _observation_error(step.get("observation"))
        if observation_error:
            environment_errors.append(f"step:{position}:{observation_error}")

    if row.get("success") is True or row.get("status") == "success":
        terminal = steps[-1] if steps and isinstance(steps[-1], dict) else {}
        terminal_info = terminal.get("environment_info")
        terminal_awm = terminal_info.get("awm") if isinstance(terminal_info, dict) else None
        if (
            row.get("success") is not True
            or row.get("status") != "success"
            or row.get("termination") != "success"
            or not isinstance(terminal_awm, dict)
            or terminal_awm.get("verifier_reward_type") != "complete"
            or terminal_awm.get("verify_execution_status") not in (None, "success")
        ):
            environment_errors.append("success_without_complete_code_verifier")

    clean = _noise_is_clean(row) and _fault_audit_is_clean(row)
    return {
        "clean": clean,
        "environment_errors": sorted(set(environment_errors)),
        "format_errors": sorted(set(format_errors)),
        "valid_attempt": clean and not environment_errors and not format_errors,
    }


def _is_verified_success(row: dict[str, Any]) -> bool:
    steps = row.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[-1], dict):
        return False
    info = steps[-1].get("environment_info")
    awm = info.get("awm") if isinstance(info, dict) else None
    return bool(
        row.get("status") == "success"
        and row.get("success") is True
        and row.get("termination") == "success"
        and isinstance(awm, dict)
        and awm.get("verifier_reward_type") == "complete"
        and awm.get("verify_execution_status") in (None, "success")
    )


def _witness_actions(row: dict[str, Any]) -> list[str]:
    diagnostics = replay_row_diagnostics(row)
    if not diagnostics["valid_attempt"]:
        raise ValueError("witness is noisy or contains format/environment errors")
    if (
        row.get("phase") != "awm_model_replay"
        or row.get("environment") != "awm"
        or row.get("status") != "success"
        or row.get("success") is not True
        or row.get("termination") != "success"
    ):
        raise ValueError("witness is not a verified AWM model-replay success")
    actions: list[str] = []
    for position, step in enumerate(row.get("steps", []), 1):
        if not isinstance(step, dict) or not isinstance(step.get("action"), str):
            raise ValueError(f"witness step {position} has no action")
        action = canonical_action(step["action"])
        if action != step["action"]:
            raise ValueError(f"witness step {position} is not canonical harness JSON")
        actions.append(action)
    if not actions or json.loads(actions[-1])["tool_name"] != "done":
        raise ValueError("witness must end with done")
    if any(json.loads(action)["tool_name"] == "done" for action in actions[:-1]):
        raise ValueError("witness contains an early done action")
    info = row["steps"][-1].get("environment_info")
    awm = info.get("awm") if isinstance(info, dict) else None
    if not isinstance(awm, dict) or awm.get("verifier_reward_type") != "complete":
        raise ValueError("witness lacks a complete code-verifier result")
    return actions


def _select_witnesses(
    rows: list[dict[str, Any]], manifest_keys: set[tuple[str, int]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], Counter[str]]:
    candidates: dict[tuple[str, int], list[tuple[tuple[int, int, int], dict[str, Any]]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for row in rows:
        key = _task_key_from_replay(row)
        if key not in manifest_keys:
            continue
        try:
            actions = _witness_actions(row)
        except (KeyError, TypeError, ValueError) as exc:
            rejected[str(exc)] += 1
            continue
        rank = (
            len(actions),
            int(row.get("generated_tokens", 0) or 0),
            int(row["_certification_source_line"]),
        )
        candidates[key].append((rank, {"row": row, "actions": actions}))
    selected = {key: min(values, key=lambda item: item[0])[1] for key, values in candidates.items()}
    return selected, rejected


def _group_replay(
    rows: list[dict[str, Any]], manifest_keys: set[tuple[str, int]]
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _task_key_from_replay(row)
        if key in manifest_keys:
            grouped[key].append(row)
    return grouped


def _policy_gate(
    rows: list[dict[str, Any]],
    *,
    min_attempts: int,
    min_success_rate: float,
    max_success_rate: float,
    max_environment_errors: int,
    max_format_errors: int,
) -> dict[str, Any]:
    environment_errors = format_errors = unclean = successes = valid_attempts = 0
    for row in rows:
        diagnostics = replay_row_diagnostics(row)
        environment_errors += int(bool(diagnostics["environment_errors"]))
        format_errors += int(bool(diagnostics["format_errors"]))
        unclean += int(not diagnostics["clean"])
        if diagnostics["valid_attempt"]:
            valid_attempts += 1
            successes += int(_is_verified_success(row))
    attempts = len(rows)
    success_rate = successes / valid_attempts if valid_attempts else 0.0
    stability_reasons = []
    if attempts < min_attempts:
        stability_reasons.append("insufficient_policy_attempts")
    if unclean:
        stability_reasons.append("policy_replay_not_clean")
    if environment_errors > max_environment_errors:
        stability_reasons.append("policy_environment_errors")
    if format_errors > max_format_errors:
        stability_reasons.append("policy_format_errors")
    learnability_reasons = []
    if valid_attempts < min_attempts:
        learnability_reasons.append("insufficient_valid_policy_attempts")
    elif success_rate < min_success_rate:
        learnability_reasons.append("policy_too_hard")
    elif success_rate > max_success_rate:
        learnability_reasons.append("policy_too_easy")
    return {
        "attempts": attempts,
        "valid_attempts": valid_attempts,
        "successes": successes,
        "success_rate": success_rate,
        "environment_error_attempts": environment_errors,
        "format_error_attempts": format_errors,
        "unclean_attempts": unclean,
        "stability_pass": not stability_reasons,
        "stability_reasons": stability_reasons,
        "learnability_pass": not learnability_reasons,
        "learnability_reasons": learnability_reasons,
    }


def _make_environment(
    factory: Callable[..., Any], task: dict[str, Any], url: str, timeout: float
) -> Any:
    return factory(task, url, timeout=timeout)


def _reset_probe(
    task: dict[str, Any],
    *,
    url: str,
    timeout: float,
    environment_factory: Callable[..., Any],
) -> dict[str, Any]:
    environment = _make_environment(environment_factory, task, url, timeout)
    try:
        result = environment.reset()
        value = json.loads(result.observation)
        tools = value.get("tools") if isinstance(value, dict) else None
        task_text = value.get("task") if isinstance(value, dict) else None
        if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
            raise RuntimeError("reset observation has no valid tools list")
        names = [tool.get("name") for tool in tools]
        if not names or not all(isinstance(name, str) and name for name in names):
            raise RuntimeError("reset observation has empty or invalid tool names")
        if len(names) != len(set(names)):
            raise RuntimeError("reset observation contains duplicate tool names")
        if not isinstance(task_text, str) or not task_text.strip():
            raise RuntimeError("reset observation has no task text")
        return {
            "schema_sha256": _json_hash(sorted(tools, key=lambda tool: str(tool.get("name")))),
            "task_sha256": hashlib.sha256(task_text.encode("utf-8")).hexdigest(),
            "tool_names": sorted(names),
        }
    finally:
        environment.close()


def _session_probe(
    task: dict[str, Any],
    actions: list[str],
    *,
    url: str,
    timeout: float,
    environment_factory: Callable[..., Any],
) -> dict[str, Any]:
    environment = _make_environment(environment_factory, task, url, timeout)
    observations: list[str] = []
    try:
        reset = environment.reset()
        observations.append(reset.observation)
        for position, action in enumerate(actions, 1):
            result = environment.step(canonical_action(action))
            observations.append(result.observation)
            awm = result.info.get("awm") if isinstance(result.info, dict) else None
            if isinstance(awm, dict) and awm.get("tool_terminal_failure"):
                return {
                    "pass": False,
                    "success": False,
                    "reason": f"step_{position}_terminal_tool_failure",
                    "observation_sha256": _json_hash(observations),
                }
            error = _observation_error(result.observation)
            if error:
                return {
                    "pass": False,
                    "success": False,
                    "reason": f"step_{position}_{error}",
                    "observation_sha256": _json_hash(observations),
                }
            if result.terminated and position != len(actions):
                return {
                    "pass": False,
                    "success": bool(result.success),
                    "reason": f"step_{position}_early_termination",
                    "observation_sha256": _json_hash(observations),
                }
        if not actions:
            return {"pass": False, "success": False, "reason": "empty_actions"}
        if not result.terminated:
            return {
                "pass": False,
                "success": bool(result.success),
                "reason": "final_action_not_terminal",
                "observation_sha256": _json_hash(observations),
            }
        return {
            "pass": True,
            "success": bool(result.success),
            "reason": "complete" if result.success else "rejected",
            "observation_sha256": _json_hash(observations),
        }
    finally:
        environment.close()


def _counterfactuals(task: dict[str, Any], actions: list[str], limit: int) -> list[dict[str, Any]]:
    values = [json.loads(action) for action in actions]
    candidates: list[dict[str, Any]] = []
    final = values[-1]
    if final["tool_name"] == "done" and isinstance(final["arguments"].get("final_answer"), str):
        replacement = dict(final)
        replacement["arguments"] = dict(final["arguments"])
        replacement["arguments"]["final_answer"] = "__AWM_CERTIFICATION_COUNTERFACTUAL__"
        changed = list(actions)
        changed[-1] = canonical_action(json.dumps(replacement, ensure_ascii=False))
        candidates.append({"kind": "replace_final_answer", "actions": changed})

    read_only = set(task.get("read_only_tools", []))
    business = [
        index
        for index, value in enumerate(values[:-1])
        if value["tool_name"] not in {"done", "verify", "list_tools"}
    ]
    ordered = [index for index in reversed(business) if values[index]["tool_name"] not in read_only]
    ordered.extend(index for index in reversed(business) if index not in ordered)
    for index in ordered:
        candidates.append(
            {
                "kind": "omit_action",
                "index": index,
                "tool_name": values[index]["tool_name"],
                "actions": actions[:index] + actions[index + 1 :],
            }
        )
    return candidates[:limit]


def _certify_live_task(
    task: dict[str, Any],
    witness: dict[str, Any] | None,
    *,
    url: str,
    timeout: float,
    reset_repeats: int,
    witness_repeats: int,
    counterfactual_candidates: int,
    require_deterministic_observations: bool,
    environment_factory: Callable[..., Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "structure": {"pass": False, "probes": []},
        "witness": {"pass": False, "probes": []},
        "verifier": {"pass": False, "no_op": None, "counterfactuals": []},
    }
    try:
        for _ in range(reset_repeats):
            result["structure"]["probes"].append(
                _reset_probe(
                    task, url=url, timeout=timeout, environment_factory=environment_factory
                )
            )
        schema_hashes = {probe["schema_sha256"] for probe in result["structure"]["probes"]}
        task_hashes = {probe["task_sha256"] for probe in result["structure"]["probes"]}
        result["structure"]["pass"] = len(schema_hashes) == 1 and len(task_hashes) == 1
        if not result["structure"]["pass"]:
            result["structure"]["reason"] = "reset_schema_or_task_changed"
            return result
    except BaseException as exc:
        result["structure"]["reason"] = f"{type(exc).__name__}: {exc}"
        return result

    if witness is None:
        result["witness"]["reason"] = "no_clean_successful_teacher_witness"
        return result
    actions = witness["actions"]
    result["witness"].update(
        source_line=witness["row"]["_certification_source_line"],
        action_count=len(actions),
        actions_sha256=_json_hash(actions),
    )
    try:
        for _ in range(witness_repeats):
            probe = _session_probe(
                task,
                actions,
                url=url,
                timeout=timeout,
                environment_factory=environment_factory,
            )
            result["witness"]["probes"].append(probe)
        hashes = {
            probe.get("observation_sha256")
            for probe in result["witness"]["probes"]
            if probe.get("observation_sha256")
        }
        result["witness"]["deterministic_observations"] = len(hashes) == 1
        result["witness"]["pass"] = all(
            probe["pass"] and probe["success"] for probe in result["witness"]["probes"]
        ) and (
            not require_deterministic_observations
            or result["witness"]["deterministic_observations"]
        )
        if not result["witness"]["pass"]:
            result["witness"]["reason"] = "witness_not_stable"
            return result
    except BaseException as exc:
        result["witness"]["reason"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        no_op = _session_probe(
            task,
            [canonical_action('{"tool_name":"done","arguments":{}}')],
            url=url,
            timeout=timeout,
            environment_factory=environment_factory,
        )
        result["verifier"]["no_op"] = no_op
        no_op_pass = no_op["pass"] and not no_op["success"]
        counterfactual_pass = False
        for candidate in _counterfactuals(task, actions, counterfactual_candidates):
            probe = _session_probe(
                task,
                candidate.pop("actions"),
                url=url,
                timeout=timeout,
                environment_factory=environment_factory,
            )
            candidate["probe"] = probe
            result["verifier"]["counterfactuals"].append(candidate)
            if probe["pass"] and not probe["success"]:
                counterfactual_pass = True
                break
        result["verifier"]["pass"] = no_op_pass and counterfactual_pass
        if not result["verifier"]["pass"]:
            result["verifier"]["reason"] = (
                "no_op_was_accepted" if not no_op_pass else "no_rejected_counterfactual"
            )
    except BaseException as exc:
        result["verifier"]["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def certify_awm_environment(
    *,
    manifest: Path,
    witness_replay: Path,
    policy_replay: Path,
    url: str,
    environment_output: Path,
    rl_output: Path,
    quarantine_output: Path,
    evidence_output: Path,
    report_output: Path,
    workers: int = 4,
    timeout: float = 180.0,
    reset_repeats: int = 3,
    witness_repeats: int = 3,
    counterfactual_candidates: int = 3,
    policy_min_attempts: int = 16,
    policy_min_success_rate: float = 0.125,
    policy_max_success_rate: float = 0.875,
    policy_max_environment_errors: int = 0,
    policy_max_format_errors: int = 0,
    require_deterministic_observations: bool = False,
    server_revision: str | None = None,
    resume: bool = False,
    environment_factory: Callable[..., Any] = AWMEnvironment,
) -> dict[str, Any]:
    for name, value in {
        "workers": workers,
        "reset_repeats": reset_repeats,
        "witness_repeats": witness_repeats,
        "counterfactual_candidates": counterfactual_candidates,
        "policy_min_attempts": policy_min_attempts,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not 0 <= policy_min_success_rate <= policy_max_success_rate <= 1:
        raise ValueError("policy success-rate bounds must satisfy 0 <= min <= max <= 1")
    if policy_max_environment_errors < 0 or policy_max_format_errors < 0:
        raise ValueError("policy error limits must be nonnegative")

    manifest = manifest.expanduser().resolve(strict=True)
    witness_replay = witness_replay.expanduser().resolve(strict=True)
    policy_replay = policy_replay.expanduser().resolve(strict=True)
    outputs = [environment_output, rl_output, quarantine_output, evidence_output, report_output]
    if len({path.expanduser().resolve() for path in outputs}) != len(outputs):
        raise ValueError("Certification output paths must be distinct")
    for path in (environment_output, rl_output, quarantine_output, report_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite certification artifact: {path}")
    if evidence_output.exists() and not resume:
        raise FileExistsError(f"Evidence exists; pass --resume or choose a new path: {evidence_output}")

    records = read_records(manifest)
    manifest_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = _task_key_from_manifest(record)
        if key in manifest_by_key:
            raise ValueError(f"Duplicate AWM scenario/task_idx in manifest: {key}")
        manifest_by_key[key] = record
    keys = set(manifest_by_key)
    witness_rows = _read_jsonl(witness_replay)
    policy_rows = _read_jsonl(policy_replay)
    witnesses, witness_rejections = _select_witnesses(witness_rows, keys)
    policy_by_key = _group_replay(policy_rows, keys)

    policy = {
        "reset_repeats": reset_repeats,
        "witness_repeats": witness_repeats,
        "counterfactual_candidates": counterfactual_candidates,
        "require_deterministic_observations": require_deterministic_observations,
        "policy_min_attempts": policy_min_attempts,
        "policy_min_success_rate": policy_min_success_rate,
        "policy_max_success_rate": policy_max_success_rate,
        "policy_max_environment_errors": policy_max_environment_errors,
        "policy_max_format_errors": policy_max_format_errors,
    }
    provenance = {
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "witness_replay": str(witness_replay),
        "witness_replay_sha256": _sha256(witness_replay),
        "policy_replay": str(policy_replay),
        "policy_replay_sha256": _sha256(policy_replay),
        "awm_url": url,
        "server_revision": server_revision,
        "trainer_awm_harness_sha256": _sha256(Path(__file__).with_name("awm.py")),
        "certifier_sha256": _sha256(Path(__file__)),
        "policy": policy,
    }
    fingerprint = _json_hash(provenance)

    decisions: dict[tuple[str, int], dict[str, Any]] = {}
    if evidence_output.exists():
        for row in _read_jsonl(evidence_output):
            if row.get("certification_fingerprint") != fingerprint:
                raise ValueError("Existing evidence was produced from different inputs or thresholds")
            key = (row.get("scenario"), row.get("task_idx"))
            if key not in keys or key in decisions:
                raise ValueError(f"Existing evidence has an unknown or duplicate task: {key}")
            row.pop("_certification_source_line", None)
            decisions[key] = row

    remaining = [key for key in manifest_by_key if key not in decisions]
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if evidence_output.exists() else "x"
    with evidence_output.open(mode, encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="awm-certify") as pool:
            completed = len(decisions)
            # Submit only one worker-sized batch at a time.  On interruption,
            # shutdown therefore waits for at most the active probes instead
            # of thousands of already-queued tasks; completed evidence remains
            # durable and can be resumed.
            for start in range(0, len(remaining), workers):
                future_to_key = {}
                for key in remaining[start : start + workers]:
                    task = manifest_by_key[key]["metadata"]["task"]
                    future = pool.submit(
                        _certify_live_task,
                        task,
                        witnesses.get(key),
                        url=url,
                        timeout=timeout,
                        reset_repeats=reset_repeats,
                        witness_repeats=witness_repeats,
                        counterfactual_candidates=counterfactual_candidates,
                        require_deterministic_observations=require_deterministic_observations,
                        environment_factory=environment_factory,
                    )
                    future_to_key[future] = key
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        live = future.result()
                    except BaseException as exc:
                        live = {
                            "structure": {
                                "pass": False,
                                "probes": [],
                                "reason": f"worker:{type(exc).__name__}: {exc}",
                            },
                            "witness": {"pass": False, "probes": []},
                            "verifier": {
                                "pass": False,
                                "no_op": None,
                                "counterfactuals": [],
                            },
                        }
                    policy_result = _policy_gate(
                        policy_by_key.get(key, []),
                        min_attempts=policy_min_attempts,
                        min_success_rate=policy_min_success_rate,
                        max_success_rate=policy_max_success_rate,
                        max_environment_errors=policy_max_environment_errors,
                        max_format_errors=policy_max_format_errors,
                    )
                    environment_pass = all(
                        live[gate]["pass"] for gate in ("structure", "witness", "verifier")
                    )
                    rl_pass = (
                        environment_pass
                        and policy_result["stability_pass"]
                        and policy_result["learnability_pass"]
                    )
                    reasons = []
                    for gate in ("structure", "witness", "verifier"):
                        if not live[gate]["pass"]:
                            reasons.append(f"{gate}:{live[gate].get('reason', 'failed')}")
                    reasons.extend(policy_result["stability_reasons"])
                    reasons.extend(policy_result["learnability_reasons"])
                    row = {
                        "schema": 1,
                        "phase": "awm_environment_certification",
                        "certification_fingerprint": fingerprint,
                        "task_id": manifest_by_key[key]["metadata"]["task"]["id"],
                        "scenario": key[0],
                        "task_idx": key[1],
                        "environment_pass": environment_pass,
                        "rl_pass": rl_pass,
                        "reasons": reasons,
                        "live": live,
                        "policy": policy_result,
                    }
                    decisions[key] = row
                    stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                        + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed += 1
                print(
                    f"AWM certification: {completed}/{len(records)}; "
                    f"environment_pass={sum(d['environment_pass'] for d in decisions.values())}; "
                    f"rl_pass={sum(d['rl_pass'] for d in decisions.values())}",
                    file=sys.stderr,
                    flush=True,
                )

    ordered_decisions = [decisions[key] for key in manifest_by_key]
    environment_records = [
        manifest_by_key[key] for key in manifest_by_key if decisions[key]["environment_pass"]
    ]
    rl_records = [manifest_by_key[key] for key in manifest_by_key if decisions[key]["rl_pass"]]
    quarantined = [decision for decision in ordered_decisions if not decision["rl_pass"]]
    write_records(environment_output, environment_records)
    write_records(rl_output, rl_records)
    _write_jsonl(quarantine_output, quarantined)

    reason_counts = Counter(reason for row in quarantined for reason in row["reasons"])
    report = {
        "schema": 1,
        "phase": "awm_environment_certification",
        "certification_fingerprint": fingerprint,
        "git_revision": _git_revision(Path(__file__).resolve().parents[2]),
        "url": url,
        "provenance": provenance,
        "tasks_input": len(records),
        "teacher_witnesses_found": len(witnesses),
        "witness_rejections": dict(sorted(witness_rejections.items())),
        "tasks_with_policy_replay": len(policy_by_key),
        "tasks_environment_certified": len(environment_records),
        "tasks_rl_certified": len(rl_records),
        "tasks_quarantined": len(quarantined),
        "quarantine_reasons": dict(sorted(reason_counts.items())),
        "outputs": {
            "environment_manifest": str(environment_output),
            "environment_manifest_sha256": _sha256(environment_output),
            "rl_manifest": str(rl_output),
            "rl_manifest_sha256": _sha256(rl_output),
            "quarantine": str(quarantine_output),
            "quarantine_sha256": _sha256(quarantine_output),
            "evidence": str(evidence_output),
            "evidence_sha256": _sha256(evidence_output),
        },
        "interpretation": {
            "environment_manifest": "Passed repeated reset, teacher witness, no-op, and counterfactual verifier gates.",
            "rl_manifest": "Also passed clean student-policy stability and intermediate-success learnability gates.",
            "database_diff_note": "The public trainer harness does not expose raw SQLite diffs; repeated code-verifier outcomes and observation hashes are recorded instead.",
            "server_revision_note": (
                "Caller-supplied AWM/OpenEnv/data revision was recorded."
                if server_revision
                else "No server revision was supplied; set AWM_CERTIFICATION_SERVER_REVISION for publication runs."
            ),
        },
    }
    atomic_json(report_output, report)
    report["report"] = str(report_output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--witness-replay", required=True, type=Path)
    parser.add_argument("--policy-replay", required=True, type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--environment-output", required=True, type=Path)
    parser.add_argument("--rl-output", required=True, type=Path)
    parser.add_argument("--quarantine-output", required=True, type=Path)
    parser.add_argument("--evidence-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--reset-repeats", type=int, default=3)
    parser.add_argument("--witness-repeats", type=int, default=3)
    parser.add_argument("--counterfactual-candidates", type=int, default=3)
    parser.add_argument("--policy-min-attempts", type=int, default=16)
    parser.add_argument("--policy-min-success-rate", type=float, default=0.125)
    parser.add_argument("--policy-max-success-rate", type=float, default=0.875)
    parser.add_argument("--policy-max-environment-errors", type=int, default=0)
    parser.add_argument("--policy-max-format-errors", type=int, default=0)
    parser.add_argument("--require-deterministic-observations", action="store_true")
    parser.add_argument(
        "--server-revision",
        help="optional immutable AWM/OpenEnv/data revision included in the evidence fingerprint",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = certify_awm_environment(
        manifest=args.manifest,
        witness_replay=args.witness_replay,
        policy_replay=args.policy_replay,
        url=args.url,
        environment_output=args.environment_output.expanduser(),
        rl_output=args.rl_output.expanduser(),
        quarantine_output=args.quarantine_output.expanduser(),
        evidence_output=args.evidence_output.expanduser(),
        report_output=args.report_output.expanduser(),
        workers=args.workers,
        timeout=args.timeout,
        reset_repeats=args.reset_repeats,
        witness_repeats=args.witness_repeats,
        counterfactual_candidates=args.counterfactual_candidates,
        policy_min_attempts=args.policy_min_attempts,
        policy_min_success_rate=args.policy_min_success_rate,
        policy_max_success_rate=args.policy_max_success_rate,
        policy_max_environment_errors=args.policy_max_environment_errors,
        policy_max_format_errors=args.policy_max_format_errors,
        require_deterministic_observations=args.require_deterministic_observations,
        server_revision=args.server_revision,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
