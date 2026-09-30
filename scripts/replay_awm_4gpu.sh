#!/usr/bin/env bash
set -euo pipefail

# Four TP=1 SGLang replicas consume four stable replay shards. AWM must be
# running already in its isolated Python environment. This script downloads
# nothing and performs no training.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_SGLANG_PYTHON_BIN="${SGLANG_PYTHON_BIN:-${TASK_PYTHON_BIN}}"

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local RL-initial HF checkpoint}"
: "${AWM_MANIFEST:?Set AWM_MANIFEST to the local context-safe candidate manifest}"
: "${AWM_URL:?Set AWM_URL to an already-running local AWM server}"

TASK_HOST="${SGLANG_HOST:-127.0.0.1}"
TASK_PORTS=(
  "${SGLANG_PORT0:-30000}"
  "${SGLANG_PORT1:-30001}"
  "${SGLANG_PORT2:-30002}"
  "${SGLANG_PORT3:-30003}"
)
TASK_MEM_FRACTION="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
TASK_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"
TASK_STARTUP_TIMEOUT="${SGLANG_STARTUP_TIMEOUT:-600}"
TASK_CONCURRENCY="${AWM_REPLAY_CONCURRENCY_PER_GPU:-24}"
TASK_REPEATS="${AWM_MODEL_REPLAY_REPEATS:-16}"
TASK_TEMPERATURE="${AWM_MODEL_REPLAY_TEMPERATURE:-1.0}"
TASK_SEED="${AWM_MODEL_REPLAY_SEED:-20260930}"
TASK_LIMIT="${AWM_MODEL_REPLAY_TASKS:-0}"
TASK_FINAL_OUTPUT="${AWM_QUAD_REPLAY_OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-student-clean-r16-4gpu-$(date -u +%Y%m%d-%H%M%S)-$$.jsonl}"

if [[ "${TASK_FINAL_OUTPUT}" != *.jsonl ]]; then
  echo "AWM_QUAD_REPLAY_OUTPUT must end in .jsonl" >&2
  exit 2
fi
for TASK_PATH in "${HF_CHECKPOINT}" "${AWM_MANIFEST}"; do
  [[ -e "${TASK_PATH}" ]] || { echo "Required local path does not exist: ${TASK_PATH}" >&2; exit 2; }
done
[[ -d "${HF_CHECKPOINT}" ]] || { echo "HF_CHECKPOINT must be a local directory" >&2; exit 2; }
[[ -f "${AWM_MANIFEST}" ]] || { echo "AWM_MANIFEST must be a JSONL file" >&2; exit 2; }
for TASK_NUMBER in "${TASK_PORTS[@]}" "${TASK_CONTEXT_LENGTH}" \
  "${TASK_STARTUP_TIMEOUT}" "${TASK_CONCURRENCY}" "${TASK_REPEATS}"; do
  case "${TASK_NUMBER}" in
    ''|*[!0-9]*|0) echo "Ports, lengths, timeout, concurrency, and repeats must be positive integers" >&2; exit 2 ;;
  esac
done
case "${TASK_LIMIT}" in
  ''|*[!0-9]*) echo "AWM_MODEL_REPLAY_TASKS must be a nonnegative integer" >&2; exit 2 ;;
esac
if [[ "$(printf '%s\n' "${TASK_PORTS[@]}" | sort -u | wc -l | tr -d ' ')" != 4 ]]; then
  echo "SGLANG_PORT0 through SGLANG_PORT3 must be distinct" >&2
  exit 2
fi

TASK_OUTPUT_STEM="${TASK_FINAL_OUTPUT%.jsonl}"
TASK_FINAL_INCIDENTS="${TASK_OUTPUT_STEM}.incidents.jsonl"
TASK_FINAL_SUMMARY="${TASK_FINAL_OUTPUT}.summary.json"
TASK_LOG_DIR="${AWM_QUAD_REPLAY_LOG_DIR:-${TASK_OUTPUT_STEM}.logs}"
TASK_GPU_CSV="${TASK_LOG_DIR}/gpu-utilization.csv"
TASK_SHARDS=()
TASK_INCIDENTS=()
TASK_SERVER_LOGS=()
TASK_OUTPUTS=("${TASK_FINAL_OUTPUT}" "${TASK_FINAL_SUMMARY}" "${TASK_FINAL_INCIDENTS}")
for TASK_INDEX in 0 1 2 3; do
  TASK_SHARDS+=("${TASK_OUTPUT_STEM}.shard${TASK_INDEX}.jsonl")
  TASK_INCIDENTS+=("${TASK_OUTPUT_STEM}.shard${TASK_INDEX}.incidents.jsonl")
  TASK_SERVER_LOGS+=("${TASK_LOG_DIR}/sglang-gpu${TASK_INDEX}.log")
  TASK_OUTPUTS+=(
    "${TASK_OUTPUT_STEM}.shard${TASK_INDEX}.jsonl"
    "${TASK_OUTPUT_STEM}.shard${TASK_INDEX}.jsonl.summary.json"
    "${TASK_OUTPUT_STEM}.shard${TASK_INDEX}.incidents.jsonl"
  )
done

for TASK_PATH in "${TASK_OUTPUTS[@]}"; do
  [[ ! -e "${TASK_PATH}" ]] || { echo "Refusing to overwrite replay artifact: ${TASK_PATH}" >&2; exit 2; }
done
[[ ! -e "${TASK_LOG_DIR}" ]] || { echo "Refusing to reuse log directory: ${TASK_LOG_DIR}" >&2; exit 2; }
mkdir -p -- "$(dirname -- "${TASK_FINAL_OUTPUT}")" "${TASK_LOG_DIR}"

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

"${TASK_PYTHON_BIN}" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"replay_awm_4gpu.sh requires exactly four visible GPUs, found {count}")
print("Validated four visible GPUs:", [torch.cuda.get_device_name(i) for i in range(count)])
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
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits \
    -l 5 >"${TASK_GPU_CSV}" 2>&1 &
  TASK_GPU_MONITOR_PID=$!
  TASK_CHILD_PIDS+=("${TASK_GPU_MONITOR_PID}")
fi

echo "Starting four independent TP=1 SGLang replicas"
TASK_SERVER_PIDS=()
for TASK_INDEX in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${TASK_INDEX}" "${TASK_SGLANG_PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${HF_CHECKPOINT}" \
    --host "${TASK_HOST}" \
    --port "${TASK_PORTS[${TASK_INDEX}]}" \
    --tp-size 1 \
    --dtype bfloat16 \
    --context-length "${TASK_CONTEXT_LENGTH}" \
    --mem-fraction-static "${TASK_MEM_FRACTION}" \
    >"${TASK_SERVER_LOGS[${TASK_INDEX}]}" 2>&1 &
  TASK_SERVER_PIDS+=("$!")
  TASK_CHILD_PIDS+=("$!")
done

wait_for_sglang() {
  local TASK_URL=$1
  local TASK_PID=$2
  local TASK_LOG=$3
  if ! "${TASK_PYTHON_BIN}" - "${TASK_URL}" "${TASK_PID}" "${TASK_STARTUP_TIMEOUT}" <<'PY'
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
    echo "SGLang failed to become ready at ${TASK_URL}; tail of ${TASK_LOG}:" >&2
    tail -n 80 "${TASK_LOG}" >&2 || true
    return 1
  fi
}

for TASK_INDEX in 0 1 2 3; do
  wait_for_sglang \
    "http://${TASK_HOST}:${TASK_PORTS[${TASK_INDEX}]}" \
    "${TASK_SERVER_PIDS[${TASK_INDEX}]}" \
    "${TASK_SERVER_LOGS[${TASK_INDEX}]}"
done

echo "Starting four disjoint replay shards; concurrency per GPU=${TASK_CONCURRENCY}; total ceiling=$((TASK_CONCURRENCY * 4))"
TASK_REPLAY_PIDS=()
for TASK_INDEX in 0 1 2 3; do
  env \
    SGLANG_URL="http://${TASK_HOST}:${TASK_PORTS[${TASK_INDEX}]}" \
    AWM_MODEL_REPLAY_OUTPUT="${TASK_SHARDS[${TASK_INDEX}]}" \
    AWM_MODEL_REPLAY_INCIDENTS="${TASK_INCIDENTS[${TASK_INDEX}]}" \
    AWM_MODEL_REPLAY_REPEATS="${TASK_REPEATS}" \
    AWM_MODEL_REPLAY_TEMPERATURE="${TASK_TEMPERATURE}" \
    AWM_MODEL_REPLAY_SEED="${TASK_SEED}" \
    AWM_MODEL_REPLAY_CONCURRENCY="${TASK_CONCURRENCY}" \
    AWM_MODEL_REPLAY_SHARD_COUNT=4 \
    AWM_MODEL_REPLAY_SHARD_INDEX="${TASK_INDEX}" \
    AWM_MODEL_REPLAY_TASKS="${TASK_LIMIT}" \
    bash "${TASK_PROJECT_DIR}/scripts/replay_awm_tasks_with_model.sh" &
  TASK_REPLAY_PIDS+=("$!")
  TASK_CHILD_PIDS+=("$!")
done

TASK_REPLAY_FAILED=0
set +e
for TASK_INDEX in 0 1 2 3; do
  wait "${TASK_REPLAY_PIDS[${TASK_INDEX}]}"
  TASK_REPLAY_STATUS=$?
  if (( TASK_REPLAY_STATUS != 0 )); then
    echo "Replay shard ${TASK_INDEX} failed with status ${TASK_REPLAY_STATUS}" >&2
    TASK_REPLAY_FAILED=1
  fi
done
set -e
if (( TASK_REPLAY_FAILED != 0 )); then
  exit 1
fi

"${TASK_PYTHON_BIN}" - \
  "${TASK_FINAL_OUTPUT}" "${TASK_FINAL_INCIDENTS}" "${TASK_FINAL_SUMMARY}" \
  "${TASK_SHARDS[@]}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

output, incidents, summary = map(Path, sys.argv[1:4])
shards = [Path(value) for value in sys.argv[4:]]
seen = set()
rows = 0
with output.open("x", encoding="utf-8") as destination:
    for shard in shards:
        with shard.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                task = row.get("task", {})
                key = (task.get("id"), row.get("attempt"))
                if key in seen:
                    raise ValueError(f"duplicate task/attempt while merging {shard}:{line_number}: {key}")
                seen.add(key)
                destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                rows += 1

with incidents.open("x", encoding="utf-8") as destination:
    for shard in shards:
        incident_path = Path(str(shard.with_suffix("")) + ".incidents.jsonl")
        if incident_path.exists():
            destination.write(incident_path.read_text(encoding="utf-8"))

summaries = [json.loads(Path(str(shard) + ".summary.json").read_text(encoding="utf-8")) for shard in shards]
status_counts = Counter()
termination_counts = Counter()
failure_counts = Counter()
for value in summaries:
    status_counts.update(value.get("status_counts", {}))
    termination_counts.update(value.get("termination_counts", {}))
    failure_counts.update(value.get("terminal_tool_failure_counts", {}))
attempts = sum(value["attempts_completed"] for value in summaries)
successes = sum(value["successes"] for value in summaries)
merged = {
    "schema": 1,
    "phase": "awm_model_replay_merged",
    "scheduler": "rolling",
    "topology": "four_tp1_replicas",
    "output": str(output.resolve()),
    "candidate_incidents": str(incidents.resolve()),
    "shards": [str(path.resolve()) for path in shards],
    "tasks": sum(value["tasks"] for value in summaries),
    "attempts_requested": sum(value["attempts_requested"] for value in summaries),
    "attempts_completed": attempts,
    "rows_merged": rows,
    "successes": successes,
    "success_rate": successes / attempts if attempts else 0.0,
    "status_counts": dict(sorted(status_counts.items())),
    "termination_counts": dict(sorted(termination_counts.items())),
    "terminal_tool_failure_counts": dict(sorted(failure_counts.items())),
    "candidate_incident_count": sum(value.get("candidate_incident_count", 0) for value in summaries),
    "candidate_task_count": sum(value.get("candidate_task_count", 0) for value in summaries),
}
summary.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(merged, ensure_ascii=False, sort_keys=True))
PY

if [[ -n "${TASK_GPU_MONITOR_PID}" ]]; then
  kill "${TASK_GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${TASK_GPU_MONITOR_PID}" 2>/dev/null || true
  "${TASK_PYTHON_BIN}" - "${TASK_GPU_CSV}" <<'PY'
import csv
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
overall = sum(means.values()) / len(means) if means else 0.0
print("Measured GPU utilization means:", means, "overall:", round(overall, 2))
PY
fi

echo "Four-GPU replay completed: ${TASK_FINAL_OUTPUT}"
echo "Per-replica SGLang logs: ${TASK_LOG_DIR}"
