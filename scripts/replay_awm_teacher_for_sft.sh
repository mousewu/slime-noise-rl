#!/usr/bin/env bash
set -euo pipefail

# Collect repeated, clean teacher demonstrations by reusing the production
# AWM model-replay harness.  SGLang and AWM must already be running locally.
# This script validates that the input manifest is train-only and scenario-
# disjoint from valid_unseen before launching any inference requests.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_TRAIN_MANIFEST:?Set AWM_TRAIN_MANIFEST to the filtered AWM train manifest}"
: "${AWM_VALID_UNSEEN_MANIFEST:?Set AWM_VALID_UNSEEN_MANIFEST to valid_unseen.jsonl}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local teacher tokenizer/checkpoint}"
: "${AWM_URL:?Set AWM_URL to an already-running local AWM server}"
: "${SGLANG_URL:?Set SGLANG_URL to an already-running SGLang teacher server}"

for TASK_PATH in "${AWM_TRAIN_MANIFEST}" "${AWM_VALID_UNSEEN_MANIFEST}" "${HF_CHECKPOINT}"; do
  if [[ ! -e "${TASK_PATH}" ]]; then
    echo "Required local path does not exist: ${TASK_PATH}" >&2
    exit 2
  fi
done
if [[ ! -f "${AWM_TRAIN_MANIFEST}" || ! -f "${AWM_VALID_UNSEEN_MANIFEST}" ]]; then
  echo "Train and valid_unseen manifests must be JSONL files" >&2
  exit 2
fi
if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "HF_CHECKPOINT must be a local tokenizer/checkpoint directory" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${TASK_PYTHON_BIN}" -c '
import json, sys

def rows(path):
    result = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("metadata", {}).get("task", {})
            if task.get("environment") != "awm" or not task.get("scenario"):
                raise SystemExit(f"{path}:{number}: expected an AWM task with scenario metadata")
            result.append(task)
    if not result:
        raise SystemExit(f"Manifest is empty: {path}")
    return result

train = rows(sys.argv[1])
valid = rows(sys.argv[2])
bad = [task["id"] for task in train if task.get("split") != "train"]
if bad:
    raise SystemExit(f"Train manifest contains non-train rows, e.g. {bad[0]}")
bad_valid = [task["id"] for task in valid if task.get("split") != "valid_unseen"]
if bad_valid:
    raise SystemExit(f"Validation manifest contains non-valid_unseen rows, e.g. {bad_valid[0]}")
valid_scenarios = {task["scenario"] for task in valid}
overlap = sorted({task["scenario"] for task in train} & valid_scenarios)
if overlap:
    raise SystemExit("Train/valid_unseen scenario overlap: " + ", ".join(overlap[:20]))
train_scenarios = {task["scenario"] for task in train}
print(f"Validated {len(train)} train tasks across {len(train_scenarios)} scenarios; "
      f"zero overlap with {len(valid_scenarios)} valid_unseen scenarios.")
' "${AWM_TRAIN_MANIFEST}" "${AWM_VALID_UNSEEN_MANIFEST}"

export AWM_MANIFEST="${AWM_TRAIN_MANIFEST}"
export AWM_MODEL_REPLAY_OUTPUT="${AWM_TEACHER_REPLAY_OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-teacher-replay-$(date -u +%Y%m%d-%H%M%S)-$$.jsonl}"
export AWM_MODEL_REPLAY_REPEATS="${AWM_TEACHER_REPEATS:-4}"
export AWM_MODEL_REPLAY_TEMPERATURE="${AWM_TEACHER_TEMPERATURE:-0.7}"
export AWM_MODEL_REPLAY_CONCURRENCY="${AWM_TEACHER_CONCURRENCY:-8}"
export AWM_MODEL_REPLAY_SEED="${AWM_TEACHER_SEED:-20260922}"

echo "Reusing the production AWM rollout harness for repeated teacher sampling."
echo "Teacher checkpoint/tokenizer: ${HF_CHECKPOINT}"
echo "Sampling: repeats=${AWM_MODEL_REPLAY_REPEATS}; temperature=${AWM_MODEL_REPLAY_TEMPERATURE}"
exec bash "${TASK_PROJECT_DIR}/scripts/replay_awm_tasks_with_model.sh"
