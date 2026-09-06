#!/usr/bin/env bash
set -euo pipefail

# Independent non-colocated launcher for Slime's official fully-async rollout.
# It intentionally does not modify scripts/train.sh or the synchronous recipe.
TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-}"
if [[ -n "${TASK_MEGATRON_LM_DIR}" ]]; then
  if [[ ! -d "${TASK_MEGATRON_LM_DIR}/megatron/training" ]]; then
    echo "MEGATRON_LM_DIR must contain megatron/training: ${TASK_MEGATRON_LM_DIR}" >&2
    exit 2
  fi
  TASK_MEGATRON_LM_DIR="$(cd -- "${TASK_MEGATRON_LM_DIR}" && pwd)"
  export MEGATRON_LM_DIR="${TASK_MEGATRON_LM_DIR}"
fi
export PYTHONPATH="${TASK_MEGATRON_LM_DIR:+${TASK_MEGATRON_LM_DIR}:}${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
exec "${TASK_PYTHON_BIN}" -m noise_rl.launch_fully_async "$@"
