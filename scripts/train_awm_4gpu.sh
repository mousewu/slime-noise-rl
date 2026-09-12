#!/usr/bin/env bash
set -euo pipefail

# AWM fully-async recipe core.  The four-GPU wrapper keeps the historical
# defaults; train_awm_8gpu.sh invokes this same checked path with a 2+6 split.
# The trainer never imports or installs OpenEnv/AWM: those NumPy-2 server
# dependencies are isolated in a separate AWM server environment. Model
# checkpoints and the manifest must be local.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${SLIME_DIR:?Set SLIME_DIR to the local Slime checkout}"
: "${MEGATRON_LM_DIR:?Set MEGATRON_LM_DIR to the local Megatron-LM source tree}"
: "${AWM_MANIFEST:?Set AWM_MANIFEST to the local AWM training JSONL manifest}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local Hugging Face checkpoint}"
: "${MEGATRON_CHECKPOINT:?Set MEGATRON_CHECKPOINT to the local Megatron checkpoint}"
: "${RUNTIME_TMPDIR:?Set RUNTIME_TMPDIR to a large local disk directory}"

TASK_TOTAL_GPUS="${AWM_TOTAL_GPUS:-4}"
TASK_ACTOR_GPUS="${AWM_ACTOR_GPUS:-2}"
TASK_ROLLOUT_GPUS="${AWM_ROLLOUT_GPUS:-2}"
TASK_TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
TASK_ENGINE_GPUS="${ENGINE_GPUS:-2}"
TASK_GPU_LABEL="${AWM_GPU_LABEL:-${TASK_TOTAL_GPUS}gpu}"
TASK_CONFIG="${CONFIG:-${TASK_PROJECT_DIR}/configs/matched_loo_awm_4gpu.yaml}"
TASK_OUTPUT="${OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-${TASK_GPU_LABEL}-$(date -u +%Y%m%d-%H%M%S)-$$}"
TASK_BATCH_SIZE="${BATCH_SIZE:-8}"
TASK_NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
TASK_NUM_ROLLOUT="${NUM_ROLLOUT:-600}"
TASK_SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
TASK_SEED="${SEED:-42}"
TASK_MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-}"
TASK_MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${TASK_MAX_CONTEXT_TOKENS:-16384}}"
TASK_CONCURRENCY="${CONCURRENCY:-}"
TASK_ENVIRONMENT_WORKERS="${ENVIRONMENT_WORKERS:-}"
TASK_INSTALL_DEPS="${INSTALL_DEPS:-1}"
TASK_START_SERVER="${START_AWM_SERVER:-0}"
TASK_AWM_HOST="${AWM_HOST:-127.0.0.1}"
TASK_AWM_PORT="${AWM_PORT:-8899}"
TASK_AWM_URL="${AWM_URL:-http://${TASK_AWM_HOST}:${TASK_AWM_PORT}}"
TASK_USE_SWANLAB="${USE_SWANLAB:-1}"
TASK_SWANLAB_MODE="${SWANLAB_MODE:-online}"
TASK_SERVER_LOG="${AWM_SERVER_LOG:-${RUNTIME_TMPDIR}/awm-server-${TASK_AWM_PORT}-$$.log}"
TASK_SERVER_PID_FILE="${AWM_SERVER_PID_FILE:-${RUNTIME_TMPDIR}/awm-server-${TASK_AWM_PORT}-$$.pid}"
TASK_AWM_SERVER_SCRIPT="${AWM_SERVER_SCRIPT:-${TASK_PROJECT_DIR}/scripts/start_awm_server.sh}"

for TASK_FLAG in "${TASK_INSTALL_DEPS}" "${TASK_START_SERVER}" "${TASK_USE_SWANLAB}"; do
  case "${TASK_FLAG}" in 0|1) ;; *) echo "Boolean options must be 0 or 1" >&2; exit 2 ;; esac
done
case "${TASK_SWANLAB_MODE}" in online|offline|local) ;; *) echo "SWANLAB_MODE must be online, offline, or local" >&2; exit 2 ;; esac
for TASK_NUMBER in \
  "${TASK_TOTAL_GPUS}" "${TASK_ACTOR_GPUS}" "${TASK_ROLLOUT_GPUS}" \
  "${TASK_TENSOR_PARALLEL}" "${TASK_ENGINE_GPUS}" "${TASK_BATCH_SIZE}" \
  "${TASK_NUM_STEPS_PER_ROLLOUT}" "${TASK_NUM_ROLLOUT}" "${TASK_SAVE_INTERVAL}" \
  "${TASK_MAX_TOKENS_PER_GPU}"; do
  case "${TASK_NUMBER}" in ''|*[!0-9]*|0) echo "GPU, batch, rollout, save, and token values must be positive integers" >&2; exit 2 ;; esac
done
for TASK_NUMBER in "${TASK_MAX_CONTEXT_TOKENS}" "${TASK_CONCURRENCY}" "${TASK_ENVIRONMENT_WORKERS}"; do
  [[ -z "${TASK_NUMBER}" ]] && continue
  case "${TASK_NUMBER}" in *[!0-9]*|0) echo "Optional context, concurrency, and environment-worker values must be positive integers" >&2; exit 2 ;; esac
done
if (( TASK_ACTOR_GPUS + TASK_ROLLOUT_GPUS != TASK_TOTAL_GPUS )); then
  echo "AWM_ACTOR_GPUS plus AWM_ROLLOUT_GPUS must equal AWM_TOTAL_GPUS" >&2
  exit 2
fi
if (( TASK_ACTOR_GPUS % TASK_TENSOR_PARALLEL != 0 )); then
  echo "AWM_ACTOR_GPUS must be divisible by TENSOR_PARALLEL" >&2
  exit 2
fi
if (( TASK_ROLLOUT_GPUS % TASK_ENGINE_GPUS != 0 )); then
  echo "AWM_ROLLOUT_GPUS must be divisible by ENGINE_GPUS" >&2
  exit 2
fi

for TASK_DIRECTORY in "${SLIME_DIR}" "${MEGATRON_LM_DIR}" "${HF_CHECKPOINT}" "${MEGATRON_CHECKPOINT}"; do
  if [[ ! -d "${TASK_DIRECTORY}" ]]; then
    echo "Required local directory does not exist: ${TASK_DIRECTORY}" >&2
    exit 2
  fi
done
for TASK_FILE in "${AWM_MANIFEST}" "${TASK_CONFIG}"; do
  if [[ ! -f "${TASK_FILE}" ]]; then
    echo "Required local file does not exist: ${TASK_FILE}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_LM_DIR}/megatron/training" ]]; then
  echo "MEGATRON_LM_DIR must contain megatron/training" >&2
  exit 2
fi
if [[ "${TASK_START_SERVER}" == 1 && ! -x "${TASK_AWM_SERVER_SCRIPT}" ]]; then
  echo "AWM_SERVER_SCRIPT must be an executable server launcher: ${TASK_AWM_SERVER_SCRIPT}" >&2
  exit 2
fi
if [[ -e "${TASK_OUTPUT}" ]]; then
  echo "OUTPUT already exists; choose a new run directory: ${TASK_OUTPUT}" >&2
  exit 2
fi

mkdir -p -- "${RUNTIME_TMPDIR}"
export TMPDIR="$(cd -- "${RUNTIME_TMPDIR}" && pwd)"
export SLIME_DIR="$(cd -- "${SLIME_DIR}" && pwd)"
export MEGATRON_LM_DIR="$(cd -- "${MEGATRON_LM_DIR}" && pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((TASK_TOTAL_GPUS - 1))")"
fi
export AWM_URL="${TASK_AWM_URL}"
export PYTHONPATH="${MEGATRON_LM_DIR}:${TASK_PROJECT_DIR}/src:${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${TASK_INSTALL_DEPS}" == 1 ]]; then
  echo "[1/6] Installing only trainer-side project and SwanLab dependencies"
  "${TASK_PYTHON_BIN}" -m pip --version >/dev/null
  "${TASK_PYTHON_BIN}" -m pip install \
    -e "${TASK_PROJECT_DIR}[tracking]"
else
  echo "[1/6] Dependency installation disabled by INSTALL_DEPS=0"
fi

echo "[2/6] Checking isolated trainer runtime and ${TASK_TOTAL_GPUS} visible GPUs"
"${TASK_PYTHON_BIN}" - "${SLIME_DIR}" "${TASK_TOTAL_GPUS}" <<'PY'
import importlib
import sys

from noise_rl.launch import verify_slime

verify_slime(sys.argv[1], "train_async.py")
for package in (
    "websockets.asyncio.client", "torch", "ray", "sglang",
    "megatron.core", "megatron.training", "transformer_engine", "flashinfer",
):
    importlib.import_module(package)
import torch
import numpy
if int(numpy.__version__.split(".", 1)[0]) >= 2:
    raise RuntimeError(
        f"Megatron trainer requires NumPy 1.x, found {numpy.__version__}. "
        "Use a dedicated trainer environment; do not install OpenEnv/AWM into it."
    )
expected_gpus = int(sys.argv[2])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpus:
    raise RuntimeError(f"Expected exactly {expected_gpus} visible GPUs, found {torch.cuda.device_count()}")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("This recipe requires BF16-capable GPUs")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(expected_gpus)])
PY

echo "[3/6] Validating local checkpoints, AWM manifest, and offline data"
"${TASK_PYTHON_BIN}" - "${HF_CHECKPOINT}" "${MEGATRON_CHECKPOINT}" "${AWM_MANIFEST}" <<'PY'
import sys
from noise_rl.data import read_records, validate_local_records
from noise_rl.preflight import validate_local_checkpoints

validate_local_checkpoints(sys.argv[1], sys.argv[2])
records = read_records(sys.argv[3])
validate_local_records(records)
if any(row["metadata"]["task"]["environment"] != "awm" for row in records):
    raise RuntimeError("AWM_MANIFEST must contain only environment=awm records")
if any(row["metadata"]["task"].get("split") != "train" for row in records):
    raise RuntimeError("AWM_MANIFEST must contain only split=train records")
print(f"Validated {len(records)} AWM training tasks")
PY

TASK_SERVER_PID=""
TASK_AWM_MIRROR_LOG=""
cleanup_awm_server() {
  if [[ -n "${TASK_SERVER_PID}" ]] && kill -0 "${TASK_SERVER_PID}" 2>/dev/null; then
    kill "${TASK_SERVER_PID}"
    wait "${TASK_SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${TASK_SERVER_PID}" ]]; then
    rm -f -- "${TASK_SERVER_PID_FILE}"
  fi
}
trap cleanup_awm_server EXIT INT TERM

if [[ "${TASK_START_SERVER}" == 1 ]]; then
  echo "[4/6] Starting an isolated local AWM server at ${TASK_AWM_URL}"
  AWM_SERVER_BACKGROUND=1 \
  AWM_SERVER_LOG="${TASK_SERVER_LOG}" \
  AWM_SERVER_PID_FILE="${TASK_SERVER_PID_FILE}" \
  AWM_HOST="${TASK_AWM_HOST}" \
  AWM_PORT="${TASK_AWM_PORT}" \
  bash "${TASK_AWM_SERVER_SCRIPT}"
  TASK_SERVER_PID="$(<"${TASK_SERVER_PID_FILE}")"
  TASK_AWM_MIRROR_LOG="${TASK_SERVER_LOG}"
else
  echo "[4/6] Using the existing AWM server at ${TASK_AWM_URL}"
  # An independently launched server has no discoverable stdout path. The
  # caller supplies AWM_SERVER_LOG when it wants its diagnostics in SwanLab.
  TASK_AWM_MIRROR_LOG="${AWM_SERVER_LOG:-}"
  if [[ -z "${TASK_AWM_MIRROR_LOG}" ]]; then
    echo "AWM server-log mirroring is unavailable for an external server; set AWM_SERVER_LOG to its file"
  fi
fi

"${TASK_PYTHON_BIN}" - "${TASK_AWM_URL}" "${TASK_SERVER_PID}" "${TASK_SERVER_LOG}" <<'PY'
import json
import os
import sys
import time
from urllib.request import urlopen

url, pid, log = sys.argv[1], sys.argv[2], sys.argv[3]
error = None
for _ in range(180):
    if pid:
        try:
            os.kill(int(pid), 0)
        except OSError as exc:
            raise RuntimeError(f"Local AWM server exited; inspect {log}") from exc
    try:
        with urlopen(url.rstrip("/") + "/stats", timeout=2) as response:
            json.load(response)
        print("AWM server is ready")
        break
    except Exception as exc:
        error = exc
        time.sleep(1)
else:
    raise RuntimeError(f"AWM server did not become ready: {error}; inspect {log}")
PY

if [[ -n "${TASK_AWM_MIRROR_LOG}" ]]; then
  export NOISE_RL_AWM_SERVER_LOG="${TASK_AWM_MIRROR_LOG}"
  export NOISE_RL_AWM_DIAGNOSTICS_PATH="${NOISE_RL_AWM_DIAGNOSTICS_PATH:-${TASK_OUTPUT}/awm_server_diagnostics.log}"
  echo "AWM server diagnostics will be mirrored to SwanLab from ${TASK_AWM_MIRROR_LOG}"
fi

TASK_TRAIN_ARGS=(
  --config "${TASK_CONFIG}"
  --data "${AWM_MANIFEST}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --megatron-checkpoint "${MEGATRON_CHECKPOINT}"
  --output "${TASK_OUTPUT}"
  --seed "${TASK_SEED}"
  --gpus "${TASK_TOTAL_GPUS}"
  --actor-gpus "${TASK_ACTOR_GPUS}"
  --rollout-gpus "${TASK_ROLLOUT_GPUS}"
  --tensor-parallel "${TASK_TENSOR_PARALLEL}"
  --engine-gpus "${TASK_ENGINE_GPUS}"
  --batch-size "${TASK_BATCH_SIZE}"
  --num-steps-per-rollout "${TASK_NUM_STEPS_PER_ROLLOUT}"
  --num-rollout "${TASK_NUM_ROLLOUT}"
  --save-interval "${TASK_SAVE_INTERVAL}"
  --max-tokens-per-gpu "${TASK_MAX_TOKENS_PER_GPU}"
)
if [[ -n "${TASK_MAX_CONTEXT_TOKENS}" ]]; then
  TASK_TRAIN_ARGS+=(--max-context-tokens "${TASK_MAX_CONTEXT_TOKENS}")
fi
if [[ -n "${TASK_CONCURRENCY}" ]]; then
  TASK_TRAIN_ARGS+=(--concurrency "${TASK_CONCURRENCY}")
fi
if [[ -n "${TASK_ENVIRONMENT_WORKERS}" ]]; then
  TASK_TRAIN_ARGS+=(--environment-workers "${TASK_ENVIRONMENT_WORKERS}")
fi
if [[ "${TASK_USE_SWANLAB}" == 1 ]]; then
  TASK_TRAIN_ARGS+=(
    --use-swanlab
    --swanlab-mode "${TASK_SWANLAB_MODE}"
    --swanlab-project "${SWANLAB_PROJECT:-agentic-noise-rl}"
    --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-$(basename -- "${TASK_OUTPUT}")}"
    --swanlab-group "${SWANLAB_GROUP:-awm-${TASK_GPU_LABEL}}"
    --swanlab-tags awm fully-async "${TASK_GPU_LABEL}" qwen3-4b
  )
fi

echo "[5/6] Validating the complete training command"
bash "${TASK_PROJECT_DIR}/scripts/train_fully_async.sh" "${TASK_TRAIN_ARGS[@]}" --dry-run

echo "[6/6] Starting ${TASK_TOTAL_GPUS}-GPU fully-async AWM training (actor=${TASK_ACTOR_GPUS}, rollout=${TASK_ROLLOUT_GPUS}, engine=${TASK_ENGINE_GPUS})"
echo "Output: ${TASK_OUTPUT}"
echo "AWM server log: ${TASK_SERVER_LOG}"
bash "${TASK_PROJECT_DIR}/scripts/train_fully_async.sh" "${TASK_TRAIN_ARGS[@]}"
