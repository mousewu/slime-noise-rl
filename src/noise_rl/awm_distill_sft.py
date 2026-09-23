"""Convert Arctic-AWM teacher replays into SFT data for a student model.

The teacher prompt is provenance only.  The student tokenizer rebuilds the
prompt from the recorded observations while reusing the teacher's verified
actions, which supports Arctic-AWM -> Qwen distillation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .agent import QwenChatProtocol
from .awm import SYSTEM_PROMPT
from .awm_context import load_local_tokenizer
from .awm_sft_data import (
    _initial_observation,
    _sha256,
    _step_actions,
    _validate_source_row,
    _write_rows,
)

DISTILLATION_DATASET = "awm_teacher_distillation_sft"


def validate_distillation_sft_records(path: str | Path) -> dict[str, Any]:
    """Validate the student-side messages emitted by this distillation builder."""
    path = Path(path).expanduser().resolve(strict=True)
    records = actions = 0
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
                    raise ValueError("messages must contain system, user, and action")
                expected = [("system", 0), ("user", 0)]
                for message in messages:
                    if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                        raise ValueError("invalid message role")
                if messages[0].get("role") != "system" or messages[1].get("role") != "user":
                    raise ValueError("messages must start with system and user")
                for message in messages[2:]:
                    if message.get("role") == "assistant":
                        if message.get("step_loss_mask") != 1:
                            raise ValueError("assistant action must have step_loss_mask=1")
                        value = json.loads(message["content"])
                        if not isinstance(value, dict) or set(value) != {"tool_name", "arguments"}:
                            raise ValueError("invalid AWM tool action")
                        actions += 1
                    elif message.get("role") == "user":
                        if message.get("step_loss_mask") != 0:
                            raise ValueError("user observation must have step_loss_mask=0")
                    else:
                        raise ValueError("messages must alternate assistant and user")
                if messages[-1].get("role") != "assistant":
                    raise ValueError("trajectory must end with assistant action")
                if not isinstance(metadata, dict) or metadata.get("dataset") != DISTILLATION_DATASET:
                    raise ValueError("unexpected distillation dataset")
                task_id = metadata.get("task_id")
                if not isinstance(task_id, str) or not task_id or task_id in task_ids:
                    raise ValueError("task_id must be nonempty and unique")
                if metadata.get("environment") != "awm" or metadata.get("split") != "train":
                    raise ValueError("distillation data must be AWM train data")
                task_ids.add(task_id)
                records += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid AWM distillation SFT data {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"AWM distillation SFT dataset is empty: {path}")
    return {"path": str(path), "dataset": DISTILLATION_DATASET, "records": records, "expert_actions": actions, "unique_tasks": len(task_ids)}


def _student_row(row, task, actions, observation, *, line, replay_path, replay_sha256, tokenizer):
    protocol = QwenChatProtocol(tokenizer)
    prompt, prompt_tokens = protocol.initial(observation, SYSTEM_PROMPT)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT, "step_loss_mask": 0},
        {"role": "user", "content": QwenChatProtocol.safe_observation(observation), "step_loss_mask": 0},
    ]
    for index, (action, next_observation) in enumerate(actions):
        messages.append({"role": "assistant", "content": action, "step_loss_mask": 1})
        if index + 1 < len(actions):
            messages.append({
                "role": "user",
                "content": QwenChatProtocol.safe_observation(next_observation),
                "step_loss_mask": 0,
            })
    return {
        "messages": messages,
        "metadata": {
            "schema": 1,
            "dataset": "awm_teacher_distillation_sft",
            "environment": "awm",
            "split": "train",
            "scenario": task["scenario"],
            "task_idx": task["task_idx"],
            "task_id": task["id"],
            "action_count": len(actions),
            "verifier": {"mode": "code", "reward_type": "complete"},
            "distillation": {
                "teacher_model": "Arctic-AWM-4B",
                "student_model": "local_checkpoint",
                "teacher_prompt_sha256": row.get("initial_prompt_sha256"),
                "student_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "student_prompt_tokens": len(prompt_tokens),
            },
            "source": {
                "replay_path": str(replay_path),
                "replay_sha256": replay_sha256,
                "replay_line": line,
                "attempt": row.get("attempt"),
            },
        },
    }


def convert(replay, output, report, student_checkpoint, max_actions=40, max_per_task=1, max_per_scenario=20):
    replay_path = Path(replay).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve()
    report_path = Path(report).expanduser().resolve()
    if output_path.exists() or report_path.exists():
        raise FileExistsError("Output or report already exists; choose new paths")
    tokenizer = load_local_tokenizer(student_checkpoint)
    replay_sha256 = _sha256(replay_path)
    candidates = []
    rejected = Counter()
    examples = []
    source_rows = 0
    with replay_path.open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            source_rows += 1
            try:
                row = json.loads(raw)
                task = _validate_source_row(row)
                actions = _step_actions(row, max_actions=max_actions)
                candidates.append((len(actions), int(row.get("generated_tokens", 10**12)), task["id"], line, row, task, actions))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                reason = str(exc) or type(exc).__name__
                rejected[reason] += 1
                if len(examples) < 30:
                    examples.append({"line": line, "reason": reason})
    selected = []
    task_counts = Counter()
    scenario_counts = Counter()
    for item in sorted(candidates):
        _, _, _, line, row, task, actions = item
        if task_counts[task["id"]] >= max_per_task:
            rejected["per_task_cap"] += 1
            continue
        if scenario_counts[task["scenario"]] >= max_per_scenario:
            rejected["per_scenario_cap"] += 1
            continue
        task_counts[task["id"]] += 1
        scenario_counts[task["scenario"]] += 1
        selected.append((line, row, task, actions))

    output_rows = []
    for line, row, task, actions in selected:
        try:
            observation, _ = _initial_observation(row, task, awm_url=None, rehydrate_timeout=180, resolver=None)
            output_rows.append(_student_row(row, task, actions, observation, line=line, replay_path=replay_path, replay_sha256=replay_sha256, tokenizer=tokenizer))
        except (TypeError, ValueError, RuntimeError) as exc:
            reason = str(exc) or type(exc).__name__
            rejected[reason] += 1
            if len(examples) < 30:
                examples.append({"line": line, "task_id": task["id"], "reason": reason})

    if not output_rows:
        raise ValueError("No distillation SFT rows survived; no output was written")
    _write_rows(output_path, output_rows)
    result = {
        "schema": 1,
        "phase": "awm_teacher_to_student_sft",
        "teacher_model": "Arctic-AWM-4B",
        "student_checkpoint": str(Path(student_checkpoint).expanduser().resolve()),
        "replay": str(replay_path),
        "source_rows": source_rows,
        "candidates": len(candidates),
        "selected_after_caps": len(selected),
        "records": len(output_rows),
        "rejected": dict(rejected.most_common()),
        "rejected_examples": examples,
        "output": str(output_path),
        "report": str(report_path),
        "policy": "Teacher actions are retained; prompts are rebuilt with the student tokenizer. Teacher and student prompt hashes are intentionally allowed to differ.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--max-per-task", type=int, default=1)
    parser.add_argument("--max-per-scenario", type=int, default=20)
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.replay, args.output, args.report, args.student_checkpoint, args.max_actions, args.max_per_task, args.max_per_scenario), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
