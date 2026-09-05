#!/usr/bin/env bash
set -euo pipefail

# End-to-end two-GPU smoke test for this project. Slime itself and its CUDA
# stack must already be installed; this script installs every project-specific
# runtime dependency without upgrading torch/SGLang/Megatron.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"
: "${SLIME_DIR:?Set SLIME_DIR to the clean, pinned Slime checkout}"

TASK_AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-1}"
TASK_ASSET_DIR="${SMOKE_ASSET_DIR:-${TASK_PROJECT_DIR}/runs/_smoke_assets}"
TASK_ALFWORLD_DATA="${ALFWORLD_DATA:-${TASK_ASSET_DIR}/alfworld}"
TASK_ALFWORLD_ROOT="${ALFWORLD_ROOT:-${TASK_ALFWORLD_DATA}/json_2.1.1}"
TASK_HF_CHECKPOINT="${HF_CHECKPOINT:-${TASK_ASSET_DIR}/Qwen3-4B-Instruct-2507}"
TASK_MEGATRON_CHECKPOINT="${MEGATRON_CHECKPOINT:-${TASK_ASSET_DIR}/Qwen3-4B-Instruct-2507_torch_dist}"
TASK_MANIFEST="${SMOKE_MANIFEST:-${TASK_ASSET_DIR}/manifests/alfworld-train-8.jsonl}"
TASK_OUTPUT="${SMOKE_OUTPUT:-${TASK_PROJECT_DIR}/runs/smoke-2gpu-$(date -u +%Y%m%d-%H%M%S)-$$}"
TASK_NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
TASK_USE_SWANLAB="${USE_SWANLAB:-1}"
TASK_SWANLAB_MODE="${SWANLAB_MODE:-offline}"
TASK_ROLLOUT_ONLY="${ROLLOUT_ONLY:-0}"

case "${TASK_AUTO_DOWNLOAD}" in 0|1) ;; *) echo "AUTO_DOWNLOAD must be 0 or 1" >&2; exit 2 ;; esac
case "${TASK_USE_SWANLAB}" in 0|1) ;; *) echo "USE_SWANLAB must be 0 or 1" >&2; exit 2 ;; esac
case "${TASK_ROLLOUT_ONLY}" in 0|1) ;; *) echo "ROLLOUT_ONLY must be 0 or 1" >&2; exit 2 ;; esac
case "${TASK_SWANLAB_MODE}" in online|offline|local) ;; *) echo "SWANLAB_MODE must be online, offline, or local" >&2; exit 2 ;; esac
case "${TASK_NUM_ROLLOUT}" in ''|*[!0-9]*) echo "NUM_ROLLOUT must be a positive integer" >&2; exit 2 ;; esac
if (( 10#${TASK_NUM_ROLLOUT} < 1 )); then
  echo "NUM_ROLLOUT must be a positive integer" >&2
  exit 2
fi
if [[ ! -d "${SLIME_DIR}" ]]; then
  echo "SLIME_DIR is not a directory: ${SLIME_DIR}" >&2
  exit 2
fi
SLIME_DIR="$(cd -- "${SLIME_DIR}" && pwd)"
export SLIME_DIR
if [[ -e "${TASK_OUTPUT}" ]]; then
  echo "SMOKE_OUTPUT already exists; choose a new path: ${TASK_OUTPUT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export ALFWORLD_DATA="${TASK_ALFWORLD_DATA}"
export PYTHONPATH="${TASK_PROJECT_DIR}/src:${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[1/7] Installing project runtime dependencies (Slime and GPU packages are left untouched)"
"${TASK_PYTHON_BIN}" -m pip --version >/dev/null
"${TASK_PYTHON_BIN}" -m pip install \
  -e "${TASK_PROJECT_DIR}[alfworld,tracking]" \
  "huggingface_hub[cli]>=0.27,<1"

echo "[2/7] Verifying the pinned Slime checkout and two visible GPUs"
"${TASK_PYTHON_BIN}" - "${SLIME_DIR}" <<'PY'
import importlib
import json
import sys

from noise_rl.launch import verify_slime

verify_slime(sys.argv[1])
missing = []
for package in ("torch", "ray", "sglang", "megatron.core", "transformer_engine", "flashinfer"):
    try:
        importlib.import_module(package)
    except ImportError:
        missing.append(package)
if missing:
    raise RuntimeError(
        "Missing Slime GPU-runtime dependencies: "
        + ", ".join(missing)
        + ". Complete Slime's installation before running this script."
    )
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError(
        f"Expected exactly 2 visible CUDA GPUs after CUDA_VISIBLE_DEVICES filtering; found {torch.cuda.device_count()}"
    )
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The two-GPU smoke recipe requires BF16-capable GPUs")
print(json.dumps({"gpus": [torch.cuda.get_device_name(i) for i in range(2)], "torch": torch.__version__}))
PY

TASK_BIN_DIR="$("${TASK_PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"

echo "[3/7] Ensuring the Qwen3-4B Hugging Face checkpoint is available"
if [[ ! -f "${TASK_HF_CHECKPOINT}/config.json" ]]; then
  if [[ "${TASK_AUTO_DOWNLOAD}" != 1 ]]; then
    echo "Missing ${TASK_HF_CHECKPOINT}/config.json and AUTO_DOWNLOAD=0" >&2
    exit 2
  fi
  if [[ ! -x "${TASK_BIN_DIR}/hf" ]]; then
    echo "The hf CLI was not installed into ${TASK_BIN_DIR}" >&2
    exit 2
  fi
  mkdir -p -- "${TASK_HF_CHECKPOINT}"
  "${TASK_BIN_DIR}/hf" download Qwen/Qwen3-4B-Instruct-2507 --local-dir "${TASK_HF_CHECKPOINT}"
fi

echo "[4/7] Ensuring the initial Megatron checkpoint is available"
if [[ ! -f "${TASK_MEGATRON_CHECKPOINT}/latest_checkpointed_iteration.txt" ]]; then
  if [[ -e "${TASK_MEGATRON_CHECKPOINT}" ]]; then
    echo "Incomplete Megatron destination already exists: ${TASK_MEGATRON_CHECKPOINT}" >&2
    echo "Move it aside and rerun; the smoke script never deletes or overwrites checkpoints." >&2
    exit 2
  fi
  bash "${TASK_PROJECT_DIR}/scripts/convert_checkpoint.sh" \
    "${TASK_HF_CHECKPOINT}" "${TASK_MEGATRON_CHECKPOINT}"
fi

echo "[5/7] Ensuring a small ALFWorld train manifest is available"
if [[ ! -d "${TASK_ALFWORLD_ROOT}/train" ]]; then
  if [[ "${TASK_AUTO_DOWNLOAD}" != 1 ]]; then
    echo "Missing ${TASK_ALFWORLD_ROOT}/train and AUTO_DOWNLOAD=0" >&2
    exit 2
  fi
  if [[ ! -x "${TASK_BIN_DIR}/alfworld-download" ]]; then
    echo "alfworld-download was not installed into ${TASK_BIN_DIR}" >&2
    exit 2
  fi
  mkdir -p -- "${TASK_ALFWORLD_DATA}"
  "${TASK_BIN_DIR}/alfworld-download" --data-dir "${TASK_ALFWORLD_DATA}"
fi
if [[ ! -f "${TASK_MANIFEST}" ]]; then
  mkdir -p -- "$(dirname -- "${TASK_MANIFEST}")"
  "${TASK_PYTHON_BIN}" -m noise_rl.cli prepare \
    --environment alfworld \
    --root "${TASK_ALFWORLD_ROOT}" \
    --split train \
    --limit 8 \
    --output "${TASK_MANIFEST}"
fi

TASK_TRAIN_ARGS=(
  --config "${TASK_PROJECT_DIR}/configs/smoke_2gpu.yaml"
  --data "${TASK_MANIFEST}"
  --hf-checkpoint "${TASK_HF_CHECKPOINT}"
  --megatron-checkpoint "${TASK_MEGATRON_CHECKPOINT}"
  --output "${TASK_OUTPUT}"
  --seed 42
  --gpus 2
  --tensor-parallel 2
  --engine-gpus 2
  --batch-size 1
  --num-rollout "${TASK_NUM_ROLLOUT}"
  --save-interval 1
  --max-tokens-per-gpu 4096
)
if [[ "${TASK_USE_SWANLAB}" == 1 ]]; then
  TASK_TRAIN_ARGS+=(
    --use-swanlab
    --swanlab-mode "${TASK_SWANLAB_MODE}"
    --swanlab-project "${SWANLAB_PROJECT:-agentic-noise-rl-smoke}"
    --swanlab-experiment-name "$(basename -- "${TASK_OUTPUT}")"
    --swanlab-group 2gpu-smoke
    --swanlab-tags smoke 2gpu qwen3-4b alfworld
  )
fi
if [[ "${TASK_ROLLOUT_ONLY}" == 1 ]]; then
  TASK_TRAIN_ARGS+=(--debug-rollout-only)
fi

echo "[6/7] Validating the complete command without creating the run directory"
bash "${TASK_PROJECT_DIR}/scripts/train.sh" "${TASK_TRAIN_ARGS[@]}" --dry-run

echo "[7/7] Starting the two-GPU online-environment smoke test"
echo "Output: ${TASK_OUTPUT}"
bash "${TASK_PROJECT_DIR}/scripts/train.sh" "${TASK_TRAIN_ARGS[@]}"

echo "Smoke test completed. Inspect ${TASK_OUTPUT}/run.json, traces/, checkpoints/, and swanlab/."
