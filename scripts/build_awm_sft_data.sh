#!/usr/bin/env bash
set -euo pipefail

# Convert only clean, code-verifier-successful AWM model replay traces into
# action-only Slime SFT data.  This does not start SGLang, Ray, or training.
# New replay logs contain initial_observation.  For older logs set AWM_SFT_AWM_URL
# so the script can reset a disposable local AWM session and verify the original
# online prompt hash/token count before admitting each trace.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_SFT_REPLAY:?Set AWM_SFT_REPLAY to a completed awm_model_replay JSONL}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local Qwen checkpoint used for replay}"

TASK_OUTPUT="${AWM_SFT_OUTPUT:-${AWM_SFT_REPLAY%.jsonl}.sft.jsonl}"
TASK_REPORT="${AWM_SFT_REPORT:-${TASK_OUTPUT%.jsonl}.report.json}"
TASK_MAX_ACTIONS="${AWM_SFT_MAX_ACTIONS:-40}"
TASK_MAX_PER_TASK="${AWM_SFT_MAX_PER_TASK:-1}"
TASK_MAX_PER_SCENARIO="${AWM_SFT_MAX_PER_SCENARIO:-20}"
TASK_REHYDRATE_TIMEOUT="${AWM_SFT_REHYDRATE_TIMEOUT:-180}"

if [[ ! -f "${AWM_SFT_REPLAY}" ]]; then
  echo "AWM_SFT_REPLAY must be an existing replay JSONL: ${AWM_SFT_REPLAY}" >&2
  exit 2
fi
if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "HF_CHECKPOINT must be an existing local model directory: ${HF_CHECKPOINT}" >&2
  exit 2
fi
if [[ -e "${TASK_OUTPUT}" || -e "${TASK_REPORT}" ]]; then
  echo "SFT output or report already exists; choose new AWM_SFT_OUTPUT/AWM_SFT_REPORT paths" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

TASK_ARGS=(
  --replay "${AWM_SFT_REPLAY}"
  --output "${TASK_OUTPUT}"
  --report "${TASK_REPORT}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --max-actions "${TASK_MAX_ACTIONS}"
  --max-per-task "${TASK_MAX_PER_TASK}"
  --max-per-scenario "${TASK_MAX_PER_SCENARIO}"
  --rehydrate-timeout "${TASK_REHYDRATE_TIMEOUT}"
)
if [[ -n "${AWM_SFT_AWM_URL:-}" ]]; then
  TASK_ARGS+=(--awm-url "${AWM_SFT_AWM_URL}")
fi
if [[ "${AWM_SFT_STRICT:-0}" == "1" ]]; then
  TASK_ARGS+=(--strict)
fi
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_sft_data "${TASK_ARGS[@]}"
