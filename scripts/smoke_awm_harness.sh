#!/usr/bin/env bash
set -euo pipefail

# Real-server AWM harness smoke. This performs no model inference, no rollout
# training, and no writes through task tools: it only resets and verifies.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_MANIFEST:?Set AWM_MANIFEST to a local AWM manifest}"
: "${AWM_URL:?Set AWM_URL to the running local AWM server URL}"

TASK_OUTPUT="${AWM_HARNESS_SMOKE_OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-harness-smoke-$(date -u +%Y%m%d-%H%M%S)-$$.jsonl}"
TASK_TASKS="${AWM_HARNESS_SMOKE_TASKS:-8}"
TASK_TIMEOUT="${AWM_HARNESS_SMOKE_TIMEOUT:-180}"
TASK_FINAL_ANSWER="${AWM_HARNESS_SMOKE_FINAL_ANSWER:-NOISE_RL_AWM_HARNESS_SMOKE}"

case "${TASK_TASKS}" in ''|*[!0-9]*|0) echo "AWM_HARNESS_SMOKE_TASKS must be a positive integer" >&2; exit 2 ;; esac
if [[ ! -f "${AWM_MANIFEST}" ]]; then
  echo "AWM_MANIFEST does not exist: ${AWM_MANIFEST}" >&2
  exit 2
fi
if [[ -e "${TASK_OUTPUT}" || -e "${TASK_OUTPUT}.summary.json" ]]; then
  echo "AWM_HARNESS_SMOKE_OUTPUT already exists: ${TASK_OUTPUT}" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_smoke \
  --data "${AWM_MANIFEST}" \
  --url "${AWM_URL}" \
  --output "${TASK_OUTPUT}" \
  --tasks "${TASK_TASKS}" \
  --timeout "${TASK_TIMEOUT}" \
  --final-answer "${TASK_FINAL_ANSWER}"
