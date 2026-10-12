#!/usr/bin/env bash
set -euo pipefail

# Paired evaluation for the HF models corresponding to rollout 100/200/300.
# Exactly one GPU must be visible. The same tasks, sampling seeds, and harness
# settings are used for all three models.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"
: "${AWM_URL:?Set AWM_URL to the running local AWM service}"
: "${AWM_EVAL_PROTOCOL_DIR:?Set AWM_EVAL_PROTOCOL_DIR created by build_awm_eval_protocol.sh}"
: "${EVAL_CHECKPOINT_ROLLOUT_100:?Set EVAL_CHECKPOINT_ROLLOUT_100 to an HF directory}"
: "${EVAL_CHECKPOINT_ROLLOUT_200:?Set EVAL_CHECKPOINT_ROLLOUT_200 to an HF directory}"
: "${EVAL_CHECKPOINT_ROLLOUT_300:?Set EVAL_CHECKPOINT_ROLLOUT_300 to an HF directory}"

TASK_PROTOCOL="${AWM_EVAL_PROTOCOL_DIR}/protocol.json"
TASK_MANIFEST="${AWM_EVAL_PROTOCOL_DIR}/replay_union.jsonl"
TASK_OUTPUT_DIR="${AWM_CHECKPOINT_EVAL_OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-rollout-100-200-300-eval-$(date -u +%Y%m%d-%H%M%S)-$$}"
TASK_REPEATS="${AWM_CHECKPOINT_EVAL_REPEATS:-4}"
TASK_SEED="${AWM_CHECKPOINT_EVAL_SEED:-20261011}"
TASK_TEMPERATURE="${AWM_CHECKPOINT_EVAL_TEMPERATURE:-0.8}"
TASK_CONCURRENCY="${AWM_REPLAY_CONCURRENCY_PER_GPU:-96}"

[[ -f "${TASK_PROTOCOL}" && -f "${TASK_MANIFEST}" ]] || {
  echo "Evaluation protocol is incomplete: ${AWM_EVAL_PROTOCOL_DIR}" >&2; exit 2;
}
[[ ! -e "${TASK_OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${TASK_OUTPUT_DIR}" >&2; exit 2; }
mkdir -p "${TASK_OUTPUT_DIR}"

TASK_LABELS=(rollout100 rollout200 rollout300)
TASK_CHECKPOINTS=(
  "${EVAL_CHECKPOINT_ROLLOUT_100}"
  "${EVAL_CHECKPOINT_ROLLOUT_200}"
  "${EVAL_CHECKPOINT_ROLLOUT_300}"
)
TASK_SUMMARY_ARGS=()
for TASK_INDEX in 0 1 2; do
  TASK_LABEL="${TASK_LABELS[${TASK_INDEX}]}"
  TASK_CHECKPOINT="${TASK_CHECKPOINTS[${TASK_INDEX}]}"
  [[ -d "${TASK_CHECKPOINT}" ]] || { echo "HF checkpoint does not exist: ${TASK_CHECKPOINT}" >&2; exit 2; }
  TASK_RUN_DIR="${TASK_OUTPUT_DIR}/${TASK_LABEL}"
  mkdir -p "${TASK_RUN_DIR}"
  echo "Evaluating ${TASK_LABEL}: ${TASK_CHECKPOINT}"
  env \
    HF_CHECKPOINT="${TASK_CHECKPOINT}" \
    AWM_MANIFEST="${TASK_MANIFEST}" \
    AWM_SINGLE_REPLAY_OUTPUT="${TASK_RUN_DIR}/replay.jsonl" \
    AWM_SINGLE_REPLAY_LOG_DIR="${TASK_RUN_DIR}/logs" \
    AWM_MODEL_REPLAY_REPEATS="${TASK_REPEATS}" \
    AWM_MODEL_REPLAY_SEED="${TASK_SEED}" \
    AWM_MODEL_REPLAY_TEMPERATURE="${TASK_TEMPERATURE}" \
    AWM_REPLAY_CONCURRENCY_PER_GPU="${TASK_CONCURRENCY}" \
    bash "${TASK_PROJECT_DIR}/scripts/replay_awm_1gpu.sh"
  TASK_SUMMARY_ARGS+=(--checkpoint "${TASK_LABEL}" "${TASK_RUN_DIR}/replay.jsonl")
done

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${TASK_PYTHON_BIN}" -m noise_rl.awm_eval_summary \
  --protocol "${TASK_PROTOCOL}" \
  "${TASK_SUMMARY_ARGS[@]}" \
  --seed "${TASK_SEED}" \
  --output-json "${TASK_OUTPUT_DIR}/summary.json" \
  --output-tsv "${TASK_OUTPUT_DIR}/summary.tsv"

echo "Rollout 100/200/300 evaluation complete: ${TASK_OUTPUT_DIR}"
echo "Main comparison: ${TASK_OUTPUT_DIR}/summary.tsv"
