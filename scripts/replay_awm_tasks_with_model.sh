#!/usr/bin/env bash
set -euo pipefail

# Fixed-policy, end-to-end AWM task replay.  This deliberately reuses an
# already-running SGLang server and does not launch Slime/Ray or update model
# weights.  Every task attempt gets a fresh server-side AWM database session.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_MANIFEST:?Set AWM_MANIFEST to a local, context-safe AWM JSONL manifest}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local Qwen tokenizer/checkpoint directory}"
: "${AWM_URL:?Set AWM_URL to an already-running local AWM server URL}"
: "${SGLANG_URL:?Set SGLANG_URL to an already-running SGLang server URL}"

TASK_CONFIG="${CONFIG:-${TASK_PROJECT_DIR}/configs/matched_loo_awm_4gpu.yaml}"
TASK_OUTPUT="${AWM_MODEL_REPLAY_OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-model-replay-$(date -u +%Y%m%d-%H%M%S)-$$.jsonl}"
TASK_INCIDENT_OUTPUT="${AWM_MODEL_REPLAY_INCIDENTS:-${TASK_OUTPUT%.jsonl}.incidents.jsonl}"
TASK_REPEATS="${AWM_MODEL_REPLAY_REPEATS:-1}"
TASK_SEED="${AWM_MODEL_REPLAY_SEED:-20260918}"
TASK_TEMPERATURE="${AWM_MODEL_REPLAY_TEMPERATURE:-0.2}"
TASK_CONCURRENCY="${AWM_MODEL_REPLAY_CONCURRENCY:-8}"
TASK_SHARD_COUNT="${AWM_MODEL_REPLAY_SHARD_COUNT:-1}"
TASK_SHARD_INDEX="${AWM_MODEL_REPLAY_SHARD_INDEX:-0}"
TASK_LIMIT="${AWM_MODEL_REPLAY_TASKS:-0}"

for TASK_PATH in "${AWM_MANIFEST}" "${HF_CHECKPOINT}" "${TASK_CONFIG}"; do
  if [[ ! -e "${TASK_PATH}" ]]; then
    echo "Required local path does not exist: ${TASK_PATH}" >&2
    exit 2
  fi
done
if [[ ! -f "${AWM_MANIFEST}" || ! -f "${TASK_CONFIG}" ]]; then
  echo "AWM_MANIFEST and CONFIG must be files" >&2
  exit 2
fi
if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "HF_CHECKPOINT must be a local directory" >&2
  exit 2
fi
for TASK_NUMBER in \
  "${TASK_REPEATS}" "${TASK_CONCURRENCY}" "${TASK_SHARD_COUNT}"; do
  case "${TASK_NUMBER}" in ''|*[!0-9]*|0) echo "Replay repeats, concurrency, and shard count must be positive integers" >&2; exit 2 ;; esac
done
for TASK_NUMBER in "${TASK_SHARD_INDEX}" "${TASK_LIMIT}"; do
  case "${TASK_NUMBER}" in ''|*[!0-9]*) echo "Replay shard index and task limit must be nonnegative integers" >&2; exit 2 ;; esac
done
if (( TASK_SHARD_INDEX >= TASK_SHARD_COUNT )); then
  echo "AWM_MODEL_REPLAY_SHARD_INDEX must be less than AWM_MODEL_REPLAY_SHARD_COUNT" >&2
  exit 2
fi
if [[ -e "${TASK_OUTPUT}" || -e "${TASK_OUTPUT}.summary.json" || -e "${TASK_INCIDENT_OUTPUT}" ]]; then
  echo "Replay output already exists; choose a new AWM_MODEL_REPLAY_OUTPUT" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

TASK_ARGS=(
  --data "${AWM_MANIFEST}"
  --output "${TASK_OUTPUT}"
  --incident-output "${TASK_INCIDENT_OUTPUT}"
  --config "${TASK_CONFIG}"
  --model "${HF_CHECKPOINT}"
  --url "${SGLANG_URL}"
  --awm-url "${AWM_URL}"
  --repeats "${TASK_REPEATS}"
  --seed "${TASK_SEED}"
  --temperature "${TASK_TEMPERATURE}"
  --concurrency "${TASK_CONCURRENCY}"
  --shard-count "${TASK_SHARD_COUNT}"
  --shard-index "${TASK_SHARD_INDEX}"
  --tasks "${TASK_LIMIT}"
)
for TASK_OPTION in MAX_TURNS MAX_TOOL_CALLS MAX_CONTEXT_TOKENS MAX_GENERATED_TOKENS MAX_TOKENS_PER_TURN; do
  TASK_VALUE="${!TASK_OPTION:-}"
  [[ -z "${TASK_VALUE}" ]] && continue
  TASK_FLAG="--$(tr '[:upper:]_' '[:lower:]-' <<<"${TASK_OPTION}")"
  TASK_ARGS+=("${TASK_FLAG}" "${TASK_VALUE}")
done

echo "Starting fixed-policy AWM replay: SGLang=${SGLANG_URL}; AWM=${AWM_URL}"
echo "No RL training, model update, Ray, or download will be started."
echo "Trace output: ${TASK_OUTPUT}"
echo "Candidate incidents: ${TASK_INCIDENT_OUTPUT}"
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_model_replay "${TASK_ARGS[@]}"
