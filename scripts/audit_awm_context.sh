#!/usr/bin/env bash
set -euo pipefail

# Query an already-running local AWM server, tokenize the exact initial prompt
# with the local Qwen tokenizer, and write a separate context-safe manifest.
# This script performs no model/data download and never imports OpenEnv.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_MANIFEST:?Set AWM_MANIFEST to the source AWM training JSONL}"
: "${AWM_CONTEXT_MANIFEST:?Set AWM_CONTEXT_MANIFEST to a new output JSONL path}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local Qwen tokenizer/checkpoint directory}"

TASK_AWM_URL="${AWM_URL:-http://127.0.0.1:8899}"
TASK_MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-16384}"
TASK_CONCURRENCY="${CONCURRENCY:-16}"
TASK_TIMEOUT="${REQUEST_TIMEOUT:-180}"
TASK_REPORT="${AWM_CONTEXT_REPORT:-${AWM_CONTEXT_MANIFEST}.report.json}"
TASK_PROGRESS_EVERY="${AWM_CONTEXT_PROGRESS_EVERY:-100}"
TASK_VERBOSE="${AWM_CONTEXT_VERBOSE:-0}"

for TASK_NUMBER in "${TASK_MAX_CONTEXT_TOKENS}" "${TASK_CONCURRENCY}" "${TASK_PROGRESS_EVERY}"; do
  case "${TASK_NUMBER}" in ''|*[!0-9]*|0) echo "Context and concurrency must be positive integers" >&2; exit 2 ;; esac
done
case "${TASK_VERBOSE}" in 0|1) ;; *) echo "AWM_CONTEXT_VERBOSE must be 0 or 1" >&2; exit 2 ;; esac
if [[ ! -f "${AWM_MANIFEST}" ]]; then
  echo "AWM_MANIFEST must be an existing JSONL file: ${AWM_MANIFEST}" >&2
  exit 2
fi
if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "HF_CHECKPOINT must be an existing local directory: ${HF_CHECKPOINT}" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "[AWM context audit] manifest=${AWM_MANIFEST}; AWM=${TASK_AWM_URL}; concurrency=${TASK_CONCURRENCY}; progress_every=${TASK_PROGRESS_EVERY}"
echo "[AWM context audit] tokenizer=${HF_CHECKPOINT}; max_context_tokens=${TASK_MAX_CONTEXT_TOKENS}; output=${AWM_CONTEXT_MANIFEST}"

TASK_ARGS=(
  --input "${AWM_MANIFEST}"
  --output "${AWM_CONTEXT_MANIFEST}"
  --report "${TASK_REPORT}"
  --model "${HF_CHECKPOINT}"
  --url "${TASK_AWM_URL}"
  --max-context-tokens "${TASK_MAX_CONTEXT_TOKENS}"
  --concurrency "${TASK_CONCURRENCY}"
  --timeout "${TASK_TIMEOUT}"
  --progress-every "${TASK_PROGRESS_EVERY}"
)
if [[ "${TASK_VERBOSE}" == 1 ]]; then
  TASK_ARGS+=(--verbose)
fi

exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_context \
  "${TASK_ARGS[@]}"
