#!/usr/bin/env bash
set -euo pipefail

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"
: "${AWM_TEACHER_REPLAY:?Set AWM_TEACHER_REPLAY to Arctic teacher replay JSONL}"
: "${STUDENT_CHECKPOINT:?Set STUDENT_CHECKPOINT to the local student checkpoint/tokenizer}"

TASK_OUTPUT="${AWM_DISTILL_SFT_OUTPUT:-${TASK_PROJECT_DIR}/data/awm/train.arctic-to-student.sft.jsonl}"
TASK_REPORT="${AWM_DISTILL_SFT_REPORT:-${TASK_OUTPUT%.jsonl}.report.json}"
for path in "${AWM_TEACHER_REPLAY}" "${STUDENT_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "Required path does not exist: ${path}" >&2; exit 2; }
done
for path in "${TASK_OUTPUT}" "${TASK_REPORT}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing to overwrite: ${path}" >&2; exit 2; }
done

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_distill_sft \
  --replay "${AWM_TEACHER_REPLAY}" \
  --student-checkpoint "${STUDENT_CHECKPOINT}" \
  --output "${TASK_OUTPUT}" \
  --report "${TASK_REPORT}" \
  --max-actions "${AWM_DISTILL_MAX_ACTIONS:-40}" \
  --max-per-task "${AWM_DISTILL_MAX_PER_TASK:-1}" \
  --max-per-scenario "${AWM_DISTILL_MAX_PER_SCENARIO:-20}"
