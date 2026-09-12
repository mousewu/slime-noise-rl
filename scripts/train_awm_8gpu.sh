#!/usr/bin/env bash
set -euo pipefail

# Eight-A100 fully-async AWM recipe.  It keeps the actor TP=2 replica on two
# GPUs and assigns the remaining six GPUs to three independent TP=2 SGLang
# rollout engines.  Seven rollout GPUs plus two actor GPUs would require nine
# physical GPUs and would violate the TP=2 engine layout.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export AWM_TOTAL_GPUS=8
export AWM_ACTOR_GPUS="${AWM_ACTOR_GPUS:-2}"
export AWM_ROLLOUT_GPUS="${AWM_ROLLOUT_GPUS:-6}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
export ENGINE_GPUS="${ENGINE_GPUS:-2}"
export AWM_GPU_LABEL="8gpu"
export CONFIG="${CONFIG:-${TASK_PROJECT_DIR}/configs/matched_loo_awm_8gpu.yaml}"

# AWM's task text plus its complete tool schemas often exceeds 8K tokens.
# Keep the rollout context and actor dynamic-packing ceiling aligned.  Use the
# context-audit script before a long run to reject any still-over-budget task.
export MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-16384}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${MAX_CONTEXT_TOKENS}}"

# train_awm_4gpu.sh divides this total across rollout engines.  48 therefore
# preserves 16 SGLang requests per engine for the 3-engine layout.
export CONCURRENCY="${CONCURRENCY:-48}"
export SWANLAB_GROUP="${SWANLAB_GROUP:-awm-8gpu}"

exec bash "${TASK_PROJECT_DIR}/scripts/train_awm_4gpu.sh"
