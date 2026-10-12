#!/usr/bin/env bash
set -euo pipefail

# High-pressure fixed-policy AWM evaluation on one GPU. One TP=1 SGLang
# server receives a rolling pool of agent trajectories. AWM must already be
# running. This script starts no training and downloads nothing.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_SGLANG_PYTHON_BIN="${SGLANG_PYTHON_BIN:-${TASK_PYTHON_BIN}}"

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to one local HF checkpoint}"
: "${AWM_MANIFEST:?Set AWM_MANIFEST to the evaluation union manifest}"
: "${AWM_URL:?Set AWM_URL to an already-running local AWM server}"

TASK_HOST="${SGLANG_HOST:-127.0.0.1}"
TASK_PORT="${SGLANG_PORT:-30000}"
TASK_MEM_FRACTION="${SGLANG_MEM_FRACTION_STATIC:-0.92}"
TASK_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"
TASK_STARTUP_TIMEOUT="${SGLANG_STARTUP_TIMEOUT:-600}"
# This is the effective agentic inference batch pressure. SGLang continuously
# batches whichever turns are ready while other trajectories wait for AWM.
TASK_CONCURRENCY="${AWM_REPLAY_CONCURRENCY_PER_GPU:-96}"
TASK_REPEATS="${AWM_MODEL_REPLAY_REPEATS:-4}"
TASK_TEMPERATURE="${AWM_MODEL_REPLAY_TEMPERATURE:-0.8}"
TASK_SEED="${AWM_MODEL_REPLAY_SEED:-20261011}"
TASK_LIMIT="${AWM_MODEL_REPLAY_TASKS:-0}"
TASK_OUTPUT="${AWM_SINGLE_REPLAY_OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-single-gpu-replay-$(date -u +%Y%m%d-%H%M%S)-$$.jsonl}"
TASK_INCIDENTS="${AWM_SINGLE_REPLAY_INCIDENTS:-${TASK_OUTPUT%.jsonl}.incidents.jsonl}"
TASK_LOG_DIR="${AWM_SINGLE_REPLAY_LOG_DIR:-${TASK_OUTPUT%.jsonl}.logs}"
TASK_SERVER_LOG="${TASK_LOG_DIR}/sglang.log"
TASK_GPU_CSV="${TASK_LOG_DIR}/gpu-utilization.csv"

[[ "${TASK_OUTPUT}" == *.jsonl ]] || { echo "AWM_SINGLE_REPLAY_OUTPUT must end in .jsonl" >&2; exit 2; }
[[ -d "${HF_CHECKPOINT}" ]] || { echo "HF_CHECKPOINT must be a local directory: ${HF_CHECKPOINT}" >&2; exit 2; }
[[ -f "${AWM_MANIFEST}" ]] || { echo "AWM_MANIFEST must be a JSONL file: ${AWM_MANIFEST}" >&2; exit 2; }
for TASK_NUMBER in "${TASK_PORT}" "${TASK_CONTEXT_LENGTH}" "${TASK_STARTUP_TIMEOUT}" "${TASK_CONCURRENCY}" "${TASK_REPEATS}"; do
  case "${TASK_NUMBER}" in
    ''|*[!0-9]*|0) echo "Port, lengths, timeout, concurrency, and repeats must be positive integers" >&2; exit 2 ;;
  esac
done
case "${TASK_LIMIT}" in
  ''|*[!0-9]*) echo "AWM_MODEL_REPLAY_TASKS must be a nonnegative integer" >&2; exit 2 ;;
esac
for TASK_PATH in "${TASK_OUTPUT}" "${TASK_OUTPUT}.summary.json" "${TASK_INCIDENTS}" "${TASK_LOG_DIR}"; do
  [[ ! -e "${TASK_PATH}" ]] || { echo "Refusing to overwrite evaluation artifact: ${TASK_PATH}" >&2; exit 2; }
done
mkdir -p -- "$(dirname -- "${TASK_OUTPUT}")" "${TASK_LOG_DIR}"

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

"${TASK_PYTHON_BIN}" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 1:
    raise SystemExit(
        f"replay_awm_1gpu.sh requires exactly one visible GPU, found {count}; "
        "set CUDA_VISIBLE_DEVICES to one physical GPU before launching"
    )
print("Validated one visible GPU:", torch.cuda.get_device_name(0))
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
  exit "${TASK_STATUS}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

TASK_GPU_MONITOR_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  TASK_MONITOR_GPU="${AWM_MONITOR_GPU_INDEX:-${CUDA_VISIBLE_DEVICES:-0}}"
  if [[ "${TASK_MONITOR_GPU}" == *,* ]]; then
    echo "AWM_MONITOR_GPU_INDEX is required when CUDA_VISIBLE_DEVICES contains multiple GPUs" >&2
    exit 2
  fi
  nvidia-smi -i "${TASK_MONITOR_GPU}" \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits -l 2 >"${TASK_GPU_CSV}" 2>&1 &
  TASK_GPU_MONITOR_PID=$!
  TASK_CHILD_PIDS+=("${TASK_GPU_MONITOR_PID}")
fi

echo "Starting one TP=1 SGLang server; rolling concurrency=${TASK_CONCURRENCY}"
"${TASK_SGLANG_PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${HF_CHECKPOINT}" \
  --host "${TASK_HOST}" \
  --port "${TASK_PORT}" \
  --tp-size 1 \
  --dtype bfloat16 \
  --context-length "${TASK_CONTEXT_LENGTH}" \
  --mem-fraction-static "${TASK_MEM_FRACTION}" \
  >"${TASK_SERVER_LOG}" 2>&1 &
TASK_SERVER_PID=$!
TASK_CHILD_PIDS+=("${TASK_SERVER_PID}")

if ! "${TASK_PYTHON_BIN}" - "http://${TASK_HOST}:${TASK_PORT}" "${TASK_SERVER_PID}" "${TASK_STARTUP_TIMEOUT}" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

base, pid, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.monotonic() + timeout
last_error = "not contacted"
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit(f"SGLang process {pid} exited before becoming ready")
    for endpoint in ("/health", "/get_model_info"):
        try:
            with urllib.request.urlopen(base + endpoint, timeout=3) as response:
                if 200 <= response.status < 300:
                    print(f"SGLang ready: {base}{endpoint}")
                    raise SystemExit(0)
        except (OSError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    time.sleep(2)
raise SystemExit(f"Timed out waiting for {base}: {last_error}")
PY
then
  echo "SGLang failed to become ready; tail of ${TASK_SERVER_LOG}:" >&2
  tail -n 100 "${TASK_SERVER_LOG}" >&2 || true
  exit 1
fi

env \
  SGLANG_URL="http://${TASK_HOST}:${TASK_PORT}" \
  AWM_MODEL_REPLAY_OUTPUT="${TASK_OUTPUT}" \
  AWM_MODEL_REPLAY_INCIDENTS="${TASK_INCIDENTS}" \
  AWM_MODEL_REPLAY_REPEATS="${TASK_REPEATS}" \
  AWM_MODEL_REPLAY_TEMPERATURE="${TASK_TEMPERATURE}" \
  AWM_MODEL_REPLAY_SEED="${TASK_SEED}" \
  AWM_MODEL_REPLAY_CONCURRENCY="${TASK_CONCURRENCY}" \
  AWM_MODEL_REPLAY_TASKS="${TASK_LIMIT}" \
  bash "${TASK_PROJECT_DIR}/scripts/replay_awm_tasks_with_model.sh"

if [[ -n "${TASK_GPU_MONITOR_PID}" ]]; then
  kill "${TASK_GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${TASK_GPU_MONITOR_PID}" 2>/dev/null || true
  TASK_GPU_MONITOR_PID=""
  "${TASK_PYTHON_BIN}" - "${TASK_GPU_CSV}" <<'PY'
import csv
import json
import sys

values = []
memory = []
with open(sys.argv[1], encoding="utf-8") as stream:
    for row in csv.reader(stream):
        if len(row) < 5:
            continue
        try:
            values.append(float(row[2].strip()))
            memory.append(float(row[3].strip()) / float(row[4].strip()))
        except (ValueError, ZeroDivisionError):
            continue
summary = {
    "samples": len(values),
    "gpu_utilization_mean": sum(values) / len(values) if values else 0.0,
    "gpu_utilization_at_least_70_fraction": sum(v >= 70 for v in values) / len(values) if values else 0.0,
    "gpu_utilization_max": max(values) if values else 0.0,
    "memory_fraction_mean": sum(memory) / len(memory) if memory else 0.0,
}
print("Single-GPU utilization:", json.dumps(summary, sort_keys=True))
PY
fi

echo "Single-GPU replay completed: ${TASK_OUTPUT}"
echo "SGLang and GPU logs: ${TASK_LOG_DIR}"
