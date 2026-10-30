#!/usr/bin/env bash
set -euo pipefail

# Evaluate one local Hugging Face checkpoint with one TP=1 SGLang instance.
# This is intentionally a thin wrapper around the matrix evaluator so its
# timeout, heartbeat, GPU telemetry, and ALFWorld runner cleanup stay identical.

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to one local Hugging Face checkpoint directory}"
: "${ALFWORLD_EVAL_DATA:?Set ALFWORLD_EVAL_DATA to held-out valid_unseen JSONL}"

TASK_METHOD="${EVAL_METHOD:-checkpoint}"
TASK_TRAIN_SEED="${EVAL_TRAIN_SEED:-0}"
TASK_GPU_ID="${EVAL_GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
TASK_OUTPUT="${ALFWORLD_EVAL_OUTPUT:-${TASK_ROOT}/runs/alfworld-single-checkpoint-eval-$(date -u +%Y%m%d-%H%M%S)-$$}"
TASK_TMP_ROOT="${TMPDIR:-/tmp}"

[[ -d "${HF_CHECKPOINT}" ]] || { echo "HF checkpoint does not exist: ${HF_CHECKPOINT}" >&2; exit 2; }
[[ -f "${ALFWORLD_EVAL_DATA}" ]] || { echo "Held-out manifest does not exist: ${ALFWORLD_EVAL_DATA}" >&2; exit 2; }
[[ "${TASK_METHOD}" != *$'\t'* && "${TASK_METHOD}" != *$'\n'* ]] || { echo "EVAL_METHOD cannot contain tabs or newlines" >&2; exit 2; }
case "${TASK_TRAIN_SEED}" in ''|*[!0-9]*) echo "EVAL_TRAIN_SEED must be a nonnegative integer" >&2; exit 2;; esac
case "${TASK_GPU_ID}" in ''|*[!0-9]*) echo "EVAL_GPU_ID must be one physical GPU index" >&2; exit 2;; esac

mkdir -p -- "${TASK_TMP_ROOT}"
TASK_CHECKPOINTS_TSV="$(mktemp "${TASK_TMP_ROOT%/}/alfworld-single-checkpoint.XXXXXX.tsv")"
cleanup_manifest() { rm -f -- "${TASK_CHECKPOINTS_TSV}"; }
trap cleanup_manifest EXIT
printf '%s\t%s\t%s\n' "${TASK_METHOD}" "${TASK_TRAIN_SEED}" "${HF_CHECKPOINT}" >"${TASK_CHECKPOINTS_TSV}"

echo "Evaluating one checkpoint with SGLang TP=1"
echo "  checkpoint: ${HF_CHECKPOINT}"
echo "  gpu: ${TASK_GPU_ID}"
echo "  conditions: ${EVAL_CONDITIONS:-matched,clean,action-only,observation-only,strong}"
echo "  output: ${TASK_OUTPUT}"

CUDA_VISIBLE_DEVICES="${TASK_GPU_ID}" \
EVAL_GPUS=1 \
ALFWORLD_EVAL_CHECKPOINTS_TSV="${TASK_CHECKPOINTS_TSV}" \
ALFWORLD_EVAL_OUTPUT="${TASK_OUTPUT}" \
EVAL_CONCURRENCY_PER_GPU="${EVAL_CONCURRENCY_PER_GPU:-64}" \
EVAL_ENVIRONMENT_PROCESSES_PER_GPU="${EVAL_ENVIRONMENT_PROCESSES_PER_GPU:-16}" \
EVAL_HEARTBEAT_INTERVAL_SECONDS="${EVAL_HEARTBEAT_INTERVAL_SECONDS:-30}" \
SGLANG_STALL_TIMEOUT_SECONDS="${SGLANG_STALL_TIMEOUT_SECONDS:-600}" \
ALLOW_LOW_GPU_UTILIZATION="${ALLOW_LOW_GPU_UTILIZATION:-1}" \
bash "${TASK_ROOT}/scripts/evaluate_alfworld_noise_matrix.sh"

