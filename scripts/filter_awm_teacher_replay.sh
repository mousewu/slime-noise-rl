#!/usr/bin/env bash
set -euo pipefail

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_MANIFEST:?Set AWM_MANIFEST to the train-only input manifest}"
: "${AWM_TEACHER_REPLAY:?Set AWM_TEACHER_REPLAY to teacher replay JSONL}"

TASK_RL_OUTPUT="${AWM_STABLE_MANIFEST:-${TASK_PROJECT_DIR}/data/awm/train.teacher-stable.jsonl}"
TASK_SFT_OUTPUT="${AWM_SFT_REPLAY_OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-teacher-success-stable.jsonl}"
TASK_REPORT="${AWM_STABLE_REPORT:-${TASK_PROJECT_DIR}/runs/awm-teacher-stable.report.json}"

for path in "${AWM_MANIFEST}" "${AWM_TEACHER_REPLAY}"; do
  [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 2; }
done
for path in "${TASK_RL_OUTPUT}" "${TASK_SFT_OUTPUT}" "${TASK_REPORT}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing to overwrite: ${path}" >&2; exit 2; }
done

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_teacher_filter \
  --manifest "${AWM_MANIFEST}" \
  --replay "${AWM_TEACHER_REPLAY}" \
  --rl-output "${TASK_RL_OUTPUT}" \
  --sft-output "${TASK_SFT_OUTPUT}" \
  --report "${TASK_REPORT}" \
  --min-attempts "${AWM_FILTER_MIN_ATTEMPTS:-4}" \
  --max-infra-failures "${AWM_FILTER_MAX_INFRA_FAILURES:-0}" \
  --max-format-errors "${AWM_FILTER_MAX_FORMAT_ERRORS:-0}" \
  ${AWM_FILTER_MAX_NON_SUCCESS:+--max-non-success "${AWM_FILTER_MAX_NON_SUCCESS}"} \
  --max-tasks-per-scenario "${AWM_FILTER_MAX_TASKS_PER_SCENARIO:-300}"
