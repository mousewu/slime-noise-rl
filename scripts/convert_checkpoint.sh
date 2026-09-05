#!/usr/bin/env bash
set -euo pipefail
: "${SLIME_DIR:?Set SLIME_DIR to a clean Slime checkout}"
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash scripts/convert_checkpoint.sh HF_MODEL_DIR NEW_MEGATRON_DIR' >&2
  exit 2
fi
TASK_HF_DIR="$1"
TASK_CKPT_DIR="$2"
if [[ -e "${TASK_CKPT_DIR}" ]]; then
  echo 'Destination already exists. Use a new path; this script never overwrites checkpoints.' >&2
  exit 2
fi
TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${TASK_PROJECT_DIR}/src:${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python -c 'import sys; from noise_rl.launch import verify_slime; verify_slime(sys.argv[1])' "${SLIME_DIR}"
python -c 'import sys; from noise_rl.preflight import validate_local_hf_checkpoint; validate_local_hf_checkpoint(sys.argv[1])' "${TASK_HF_DIR}"
source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"
exec python "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
  "${MODEL_ARGS[@]}" --hf-checkpoint "${TASK_HF_DIR}" --save "${TASK_CKPT_DIR}" --bf16
