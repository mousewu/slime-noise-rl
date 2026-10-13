#!/usr/bin/env bash
set -euo pipefail

# Eight-GPU synchronous/colocated ALFWorld recipe optimized for aggregate GPU
# occupancy. During rollout every GPU hosts one TP=1 SGLang engine; during the
# optimizer phase all eight GPUs form a TP=2, DP=4 Megatron actor. This avoids
# permanently reserving idle actor or rollout GPUs as in fully-async training.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${SLIME_DIR:?Set SLIME_DIR to the local clean Slime checkout}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local tokenizer/HF checkpoint}"
: "${MEGATRON_CHECKPOINT:?Set MEGATRON_CHECKPOINT to the local SFT or RL-initial Megatron checkpoint}"
: "${ALFWORLD_MANIFEST:?Set ALFWORLD_MANIFEST to the local ALFWorld train JSONL}"

TASK_CONFIG="${CONFIG:-${TASK_PROJECT_DIR}/configs/matched_loo.yaml}"
TASK_SEED="${SEED:-42}"
TASK_BATCH_SIZE="${BATCH_SIZE:-16}"
TASK_CONCURRENCY="${CONCURRENCY:-128}"
TASK_ENV_PROCESSES="${ENVIRONMENT_PROCESSES:-32}"
TASK_ENV_RECYCLE="${ENVIRONMENT_RECYCLE_EPISODES:-32}"
TASK_NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
TASK_SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
TASK_MAX_CONTEXT="${MAX_CONTEXT_TOKENS:-8192}"
TASK_MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
TASK_LR="${LR:-1e-6}"
TASK_RUN_NAME="${RUN_NAME:-alfworld-$(basename "${TASK_CONFIG}" .yaml)-colocate8-s${TASK_SEED}}"
TASK_OUTPUT="${OUTPUT_DIR:-${TASK_PROJECT_DIR}/runs/${TASK_RUN_NAME}}"
TASK_GPU_LOG="${GPU_UTILIZATION_LOG:-${TASK_OUTPUT}.gpu-utilization.csv}"

for TASK_PATH in "${TASK_CONFIG}" "${ALFWORLD_MANIFEST}" "${HF_CHECKPOINT}" "${MEGATRON_CHECKPOINT}"; do
  [[ -e "${TASK_PATH}" ]] || { echo "Required local path does not exist: ${TASK_PATH}" >&2; exit 2; }
done
for TASK_NUMBER in "${TASK_BATCH_SIZE}" "${TASK_CONCURRENCY}" "${TASK_NUM_ROLLOUT}" \
  "${TASK_SAVE_INTERVAL}" "${TASK_MAX_CONTEXT}" "${TASK_MAX_TOKENS_PER_GPU}"; do
  case "${TASK_NUMBER}" in
    ''|*[!0-9]*|0) echo "Batch, concurrency, rollout, save, and token limits must be positive integers" >&2; exit 2 ;;
  esac
done
for TASK_NUMBER in "${TASK_ENV_PROCESSES}" "${TASK_ENV_RECYCLE}"; do
  case "${TASK_NUMBER}" in
    ''|*[!0-9]*) echo "Environment process settings must be nonnegative integers" >&2; exit 2 ;;
  esac
done
if (( TASK_BATCH_SIZE * 8 < TASK_CONCURRENCY )); then
  echo "batch-size * group-size must cover concurrency; got $((TASK_BATCH_SIZE * 8)) < ${TASK_CONCURRENCY}" >&2
  exit 2
fi
if (( TASK_CONCURRENCY % 8 != 0 )); then
  echo "CONCURRENCY must be divisible by eight single-GPU rollout engines" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

"${TASK_PYTHON_BIN}" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"high-util ALFWorld recipe requires exactly 8 visible GPUs, found {count}")
print("Validated eight visible GPUs:", [torch.cuda.get_device_name(i) for i in range(count)])
PY

TASK_CHILD_PIDS=()
cleanup() {
  local TASK_STATUS=$?
  trap - EXIT
  for TASK_PID in "${TASK_CHILD_PIDS[@]:-}"; do
    if [[ -n "${TASK_PID}" ]] && kill -0 "${TASK_PID}" 2>/dev/null; then
      kill "${TASK_PID}" 2>/dev/null || true
    fi
  done
  for TASK_PID in "${TASK_CHILD_PIDS[@]:-}"; do
    [[ -n "${TASK_PID}" ]] && wait "${TASK_PID}" 2>/dev/null || true
  done
  if [[ -s "${TASK_GPU_LOG}" ]]; then
    "${TASK_PYTHON_BIN}" - "${TASK_GPU_LOG}" <<'PY' || true
import csv
import json
import sys
from collections import defaultdict

values = defaultdict(list)
with open(sys.argv[1], encoding="utf-8") as stream:
    for row in csv.reader(stream):
        if len(row) < 3:
            continue
        try:
            values[int(row[1].strip())].append(float(row[2].strip()))
        except ValueError:
            continue
means = {gpu: sum(items) / len(items) for gpu, items in sorted(values.items()) if items}
all_samples = [sample for samples in values.values() for sample in samples]
summary = {
    "per_gpu_mean": means,
    "overall_sample_mean": sum(all_samples) / len(all_samples) if all_samples else 0.0,
    "samples_at_least_70_fraction": sum(v >= 70 for v in all_samples) / len(all_samples) if all_samples else 0.0,
}
print("ALFWorld GPU utilization:", json.dumps(summary, sort_keys=True))
PY
  fi
  exit "${TASK_STATUS}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if command -v nvidia-smi >/dev/null 2>&1; then
  mkdir -p -- "$(dirname -- "${TASK_GPU_LOG}")"
  TASK_MONITOR_GPUS="${AWM_MONITOR_GPU_INDEX:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
  nvidia-smi -i "${TASK_MONITOR_GPUS}" \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits -l 5 >"${TASK_GPU_LOG}" 2>&1 &
  TASK_CHILD_PIDS+=("$!")
fi

TASK_ARGS=(
  --config "${TASK_CONFIG}"
  --data "${ALFWORLD_MANIFEST}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --megatron-checkpoint "${MEGATRON_CHECKPOINT}"
  --output "${TASK_OUTPUT}"
  --seed "${TASK_SEED}"
  --gpus 8
  --tensor-parallel 2
  --engine-gpus 1
  --batch-size "${TASK_BATCH_SIZE}"
  --num-steps-per-rollout 1
  --num-rollout "${TASK_NUM_ROLLOUT}"
  --save-interval "${TASK_SAVE_INTERVAL}"
  --max-context-tokens "${TASK_MAX_CONTEXT}"
  --max-tokens-per-gpu "${TASK_MAX_TOKENS_PER_GPU}"
  --concurrency "${TASK_CONCURRENCY}"
  --environment-processes "${TASK_ENV_PROCESSES}"
  --environment-recycle-episodes "${TASK_ENV_RECYCLE}"
  --lr "${TASK_LR}"
)
if [[ "${USE_SWANLAB:-1}" == "1" ]]; then
  TASK_ARGS+=(
    --use-swanlab
    --swanlab-project "${SWANLAB_PROJECT:-agentic-noise-rl}"
    --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-${TASK_RUN_NAME}}"
    --swanlab-group "${SWANLAB_GROUP:-alfworld-noise-rl}"
    --swanlab-tags alfworld noise-rl colocate 8gpu high-util "seed-${TASK_SEED}"
    --swanlab-mode "${SWANLAB_MODE:-online}"
  )
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  TASK_ARGS+=(--dry-run)
fi

echo "Starting ALFWorld colocated training: batch=${TASK_BATCH_SIZE}, trajectories=$((TASK_BATCH_SIZE * 8)), concurrency=${TASK_CONCURRENCY}"
echo "Topology: actor TP=2/DP=4 on all 8 GPUs; rollout=8 independent TP=1 engines"
bash "${TASK_PROJECT_DIR}/scripts/train.sh" "${TASK_ARGS[@]}"
