#!/usr/bin/env bash
set -euo pipefail

# Eight-GPU fully-async ALFWorld recipe aimed at high aggregate GPU occupancy.
# GPU 0-1: one Megatron actor (TP=2).
# GPU 2-7: six independent TP=1 SGLang engines.
#
# The historical recipe used three TP=2 rollout engines and about 64 in-flight
# episodes.  Qwen3-4B fits on one A100-80GB for inference, so TP=1 doubles the
# number of independent engines and the default concurrency is raised to 192
# (32 in-flight episodes per engine).

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${SLIME_DIR:?Set SLIME_DIR to the local clean Slime checkout}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local tokenizer/HF checkpoint}"
: "${MEGATRON_CHECKPOINT:?Set MEGATRON_CHECKPOINT to the local SFT or RL-initial Megatron checkpoint}"
: "${ALFWORLD_MANIFEST:?Set ALFWORLD_MANIFEST to the local ALFWorld train JSONL}"

TASK_CONFIG="${CONFIG:-${TASK_PROJECT_DIR}/configs/matched_loo_fully_async.yaml}"
TASK_SEED="${SEED:-42}"
TASK_ACTOR_GPUS="${ACTOR_GPUS:-2}"
TASK_ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"
TASK_TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
TASK_ENGINE_GPUS="${ENGINE_GPUS:-1}"
TASK_BATCH_SIZE="${BATCH_SIZE:-64}"
TASK_NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
TASK_CONCURRENCY="${CONCURRENCY:-192}"
TASK_ENV_PROCESSES="${ENVIRONMENT_PROCESSES:-64}"
TASK_ENV_RECYCLE="${ENVIRONMENT_RECYCLE_EPISODES:-32}"
TASK_NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
TASK_SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
TASK_MAX_CONTEXT="${MAX_CONTEXT_TOKENS:-8192}"
TASK_MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
TASK_SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION_STATIC:-0.55}"
TASK_LR="${LR:-1e-6}"
TASK_KL_COEF="${KL_COEF:-0}"
TASK_ENTROPY_COEF="${ENTROPY_COEF:-0}"
TASK_EPS_CLIP="${EPS_CLIP:-0.2}"
TASK_EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-${TASK_EPS_CLIP}}"
TASK_USE_TIS="${USE_TIS:-0}"
TASK_TIS_CLIP="${TIS_CLIP:-2.0}"
TASK_TIS_CLIP_LOW="${TIS_CLIP_LOW:-0.0}"
TASK_GRPO_STD_NORMALIZATION="${GRPO_STD_NORMALIZATION:-0}"
TASK_DYNAMIC_FILTER="${DYNAMIC_SAMPLING_FILTER_PATH:-}"
TASK_RUN_NAME="${RUN_NAME:-alfworld-matched-loo-async8-tp1-c${TASK_CONCURRENCY}-s${TASK_SEED}}"
TASK_OUTPUT="${OUTPUT_DIR:-${TASK_PROJECT_DIR}/runs/${TASK_RUN_NAME}}"
TASK_GPU_LOG="${GPU_UTILIZATION_LOG:-${TASK_OUTPUT}.gpu-utilization.csv}"

for TASK_PATH in "${TASK_CONFIG}" "${ALFWORLD_MANIFEST}" "${HF_CHECKPOINT}" "${MEGATRON_CHECKPOINT}"; do
  [[ -e "${TASK_PATH}" ]] || { echo "Required local path does not exist: ${TASK_PATH}" >&2; exit 2; }
done
for TASK_NUMBER in "${TASK_ACTOR_GPUS}" "${TASK_ROLLOUT_GPUS}" "${TASK_TENSOR_PARALLEL}" \
  "${TASK_ENGINE_GPUS}" "${TASK_BATCH_SIZE}" "${TASK_NUM_STEPS_PER_ROLLOUT}" \
  "${TASK_CONCURRENCY}" "${TASK_ENV_PROCESSES}" \
  "${TASK_NUM_ROLLOUT}" "${TASK_SAVE_INTERVAL}" "${TASK_MAX_CONTEXT}" "${TASK_MAX_TOKENS_PER_GPU}"; do
  case "${TASK_NUMBER}" in
    ''|*[!0-9]*|0) echo "Batch, concurrency, process, rollout, save, and token settings must be positive integers" >&2; exit 2 ;;
  esac
done
if (( TASK_ACTOR_GPUS + TASK_ROLLOUT_GPUS != 8 )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS must equal 8" >&2
  exit 2
fi
if (( TASK_ACTOR_GPUS % TASK_TENSOR_PARALLEL != 0 )); then
  echo "ACTOR_GPUS must be divisible by TENSOR_PARALLEL" >&2
  exit 2
fi
if (( TASK_ROLLOUT_GPUS % TASK_ENGINE_GPUS != 0 )); then
  echo "ROLLOUT_GPUS must be divisible by ENGINE_GPUS" >&2
  exit 2
fi
case "${TASK_ENV_RECYCLE}" in
  ''|*[!0-9]*) echo "ENVIRONMENT_RECYCLE_EPISODES must be a nonnegative integer" >&2; exit 2 ;;
esac
for TASK_BOOLEAN in "${TASK_USE_TIS}" "${TASK_GRPO_STD_NORMALIZATION}"; do
  case "${TASK_BOOLEAN}" in
    0|1) ;;
    *) echo "USE_TIS and GRPO_STD_NORMALIZATION must be 0 or 1" >&2; exit 2 ;;
  esac
done
if (( TASK_BATCH_SIZE * 8 < TASK_CONCURRENCY )); then
  echo "BATCH_SIZE * group_size must cover CONCURRENCY; got $((TASK_BATCH_SIZE * 8)) < ${TASK_CONCURRENCY}" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

"${TASK_PYTHON_BIN}" - "${TASK_SGLANG_MEM_FRACTION}" <<'PY'
import sys

import torch

mem_fraction = float(sys.argv[1])
if not 0 < mem_fraction < 1:
    raise SystemExit(f"SGLANG_MEM_FRACTION_STATIC must be between 0 and 1, found {mem_fraction}")

count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"async ALFWorld recipe requires exactly 8 visible GPUs, found {count}")
print("Validated eight visible GPUs:", [torch.cuda.get_device_name(i) for i in range(count)])
PY

TASK_MONITOR_PID=""
cleanup() {
  local TASK_STATUS=$?
  trap - EXIT
  if [[ -n "${TASK_MONITOR_PID}" ]] && kill -0 "${TASK_MONITOR_PID}" 2>/dev/null; then
    kill "${TASK_MONITOR_PID}" 2>/dev/null || true
    wait "${TASK_MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -s "${TASK_GPU_LOG}" ]]; then
    "${TASK_PYTHON_BIN}" - "${TASK_GPU_LOG}" <<'PY' || true
import csv, json, sys
from collections import defaultdict

values = defaultdict(list)
with open(sys.argv[1], encoding="utf-8") as stream:
    for row in csv.reader(stream):
        if len(row) < 3:
            continue
        try:
            values[int(row[1].strip())].append(float(row[2].strip()))
        except ValueError:
            pass
means = {gpu: sum(xs) / len(xs) for gpu, xs in sorted(values.items()) if xs}
samples = [x for xs in values.values() for x in xs]
print("ALFWorld async GPU utilization:", json.dumps({
    "per_gpu_mean": means,
    "overall_sample_mean": sum(samples) / len(samples) if samples else 0.0,
    "samples_at_least_70_fraction": sum(x >= 70 for x in samples) / len(samples) if samples else 0.0,
}, sort_keys=True))
PY
  fi
  exit "${TASK_STATUS}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if command -v nvidia-smi >/dev/null 2>&1; then
  mkdir -p -- "$(dirname -- "${TASK_GPU_LOG}")"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits -l 5 >"${TASK_GPU_LOG}" 2>&1 &
  TASK_MONITOR_PID=$!
fi

TASK_ARGS=(
  --config "${TASK_CONFIG}"
  --data "${ALFWORLD_MANIFEST}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --megatron-checkpoint "${MEGATRON_CHECKPOINT}"
  --output "${TASK_OUTPUT}"
  --seed "${TASK_SEED}"
  --gpus 8
  --actor-gpus "${TASK_ACTOR_GPUS}"
  --rollout-gpus "${TASK_ROLLOUT_GPUS}"
  --tensor-parallel "${TASK_TENSOR_PARALLEL}"
  --engine-gpus "${TASK_ENGINE_GPUS}"
  --batch-size "${TASK_BATCH_SIZE}"
  --num-steps-per-rollout "${TASK_NUM_STEPS_PER_ROLLOUT}"
  --num-rollout "${TASK_NUM_ROLLOUT}"
  --save-interval "${TASK_SAVE_INTERVAL}"
  --max-context-tokens "${TASK_MAX_CONTEXT}"
  --max-tokens-per-gpu "${TASK_MAX_TOKENS_PER_GPU}"
  --sglang-mem-fraction-static "${TASK_SGLANG_MEM_FRACTION}"
  --concurrency "${TASK_CONCURRENCY}"
  --environment-processes "${TASK_ENV_PROCESSES}"
  --environment-recycle-episodes "${TASK_ENV_RECYCLE}"
  --lr "${TASK_LR}"
  --kl-coef "${TASK_KL_COEF}"
  --entropy-coef "${TASK_ENTROPY_COEF}"
  --eps-clip "${TASK_EPS_CLIP}"
  --eps-clip-high "${TASK_EPS_CLIP_HIGH}"
  --tis-clip "${TASK_TIS_CLIP}"
  --tis-clip-low "${TASK_TIS_CLIP_LOW}"
)
if [[ "${TASK_USE_TIS}" == "1" ]]; then
  TASK_ARGS+=(--use-tis)
fi
if [[ "${TASK_GRPO_STD_NORMALIZATION}" == "1" ]]; then
  TASK_ARGS+=(--grpo-std-normalization)
fi
if [[ -n "${TASK_DYNAMIC_FILTER}" ]]; then
  TASK_ARGS+=(--dynamic-sampling-filter-path "${TASK_DYNAMIC_FILTER}")
fi
if [[ "${USE_SWANLAB:-1}" == "1" ]]; then
  TASK_ARGS+=(
    --use-swanlab
    --swanlab-project "${SWANLAB_PROJECT:-agentic-noise-rl}"
    --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-${TASK_RUN_NAME}}"
    --swanlab-group "${SWANLAB_GROUP:-alfworld-noise-rl}"
    --swanlab-tags alfworld noise-rl fully-async 8gpu tp1-rollout high-util "seed-${TASK_SEED}"
    --swanlab-mode "${SWANLAB_MODE:-online}"
  )
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  TASK_ARGS+=(--dry-run)
fi

echo "Starting ALFWorld fully-async training: batch=${TASK_BATCH_SIZE}, trajectories=$((TASK_BATCH_SIZE * 8)), concurrency=${TASK_CONCURRENCY}"
echo "Topology: actor=${TASK_ACTOR_GPUS} GPUs (TP=${TASK_TENSOR_PARALLEL}); rollout=${TASK_ROLLOUT_GPUS} GPUs (${TASK_ENGINE_GPUS} GPU/engine); environment_processes=${TASK_ENV_PROCESSES}; sglang_mem_fraction_static=${TASK_SGLANG_MEM_FRACTION}"
echo "Stability: lr=${TASK_LR}; kl=${TASK_KL_COEF}; entropy=${TASK_ENTROPY_COEF}; clip=${TASK_EPS_CLIP}/${TASK_EPS_CLIP_HIGH}; tis=${TASK_USE_TIS} [${TASK_TIS_CLIP_LOW}, ${TASK_TIS_CLIP}]; grpo_std_normalization=${TASK_GRPO_STD_NORMALIZATION}; dynamic_filter=${TASK_DYNAMIC_FILTER:-off}"
bash "${TASK_PROJECT_DIR}/scripts/train_fully_async.sh" "${TASK_ARGS[@]}"
