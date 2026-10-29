#!/usr/bin/env bash
set -euo pipefail

# Evaluate multiple local HF checkpoints on held-out ALFWorld under a fixed
# noise matrix. Each GPU runs a different checkpoint in an independent TP=1
# SGLang replica and evaluates the complete held-out manifest. No model or
# dataset is downloaded.

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${PYTHON_BIN:-python}"
TASK_SGLANG_PYTHON="${SGLANG_PYTHON_BIN:-${TASK_PYTHON}}"

: "${ALFWORLD_EVAL_DATA:?Set ALFWORLD_EVAL_DATA to held-out valid_unseen JSONL}"
: "${ALFWORLD_EVAL_CHECKPOINTS_TSV:?Set ALFWORLD_EVAL_CHECKPOINTS_TSV to method<TAB>seed<TAB>local_hf_path}"

TASK_CONFIG="${ALFWORLD_EVAL_CONFIG:-${TASK_ROOT}/configs/matched_loo_fully_async.yaml}"
TASK_OUTPUT="${ALFWORLD_EVAL_OUTPUT:-${TASK_ROOT}/runs/alfworld-noise-eval-$(date -u +%Y%m%d-%H%M%S)-$$}"
TASK_GPU_COUNT="${EVAL_GPUS:-4}"
TASK_REPEATS="${EVAL_REPEATS:-4}"
TASK_SEED="${EVAL_SEED:-20261027}"
TASK_TEMPERATURE="${EVAL_TEMPERATURE:-0.8}"
TASK_CONCURRENCY="${EVAL_CONCURRENCY_PER_GPU:-96}"
TASK_ENV_PROCESSES="${EVAL_ENVIRONMENT_PROCESSES_PER_GPU:-32}"
TASK_CONTEXT="${SGLANG_CONTEXT_LENGTH:-8192}"
TASK_MEM_FRACTION="${SGLANG_MEM_FRACTION_STATIC:-0.80}"
TASK_HOST="${SGLANG_HOST:-127.0.0.1}"
TASK_PORT_BASE="${SGLANG_PORT_BASE:-31000}"
TASK_STARTUP_TIMEOUT="${SGLANG_STARTUP_TIMEOUT:-600}"
TASK_CLEANUP_TIMEOUT="${SGLANG_CLEANUP_TIMEOUT:-30}"
TASK_CLEANUP_PAUSE="${CHECKPOINT_CLEANUP_SECONDS:-5}"
TASK_STARTUP_STAGGER="${SGLANG_STARTUP_STAGGER_SECONDS:-20}"
TASK_DROP_CACHE="${SGLANG_DROP_CACHE_AFTER_LOAD:-1}"
TASK_UTIL_MIN="${GPU_UTILIZATION_MIN:-70}"
TASK_UTIL_INTERVAL="${GPU_UTILIZATION_INTERVAL_SECONDS:-2}"
TASK_ALLOW_LOW_UTIL="${ALLOW_LOW_GPU_UTILIZATION:-0}"
TASK_CONDITIONS="${EVAL_CONDITIONS:-matched,clean,action-only,observation-only,strong}"
TASK_LIMIT="${EVAL_TASK_LIMIT:-0}"

for TASK_PATH in "${ALFWORLD_EVAL_DATA}" "${ALFWORLD_EVAL_CHECKPOINTS_TSV}" "${TASK_CONFIG}"; do
  [[ -f "${TASK_PATH}" ]] || { echo "Required local file does not exist: ${TASK_PATH}" >&2; exit 2; }
done
[[ ! -e "${TASK_OUTPUT}" ]] || { echo "Refusing to reuse output directory: ${TASK_OUTPUT}" >&2; exit 2; }
for TASK_VALUE in "${TASK_GPU_COUNT}" "${TASK_REPEATS}" "${TASK_CONCURRENCY}" \
  "${TASK_ENV_PROCESSES}" "${TASK_CONTEXT}" "${TASK_PORT_BASE}" "${TASK_STARTUP_TIMEOUT}" \
  "${TASK_CLEANUP_TIMEOUT}"; do
  case "${TASK_VALUE}" in ''|*[!0-9]*|0) echo "GPU count, repeats, concurrency, process count, context and ports must be positive integers" >&2; exit 2;; esac
done
case "${TASK_LIMIT}" in ''|*[!0-9]*) echo "EVAL_TASK_LIMIT must be a nonnegative integer" >&2; exit 2;; esac
case "${TASK_CLEANUP_PAUSE}" in ''|*[!0-9]*) echo "CHECKPOINT_CLEANUP_SECONDS must be a nonnegative integer" >&2; exit 2;; esac
case "${TASK_STARTUP_STAGGER}" in ''|*[!0-9]*) echo "SGLANG_STARTUP_STAGGER_SECONDS must be a nonnegative integer" >&2; exit 2;; esac
case "${TASK_DROP_CACHE}" in 0|1) ;; *) echo "SGLANG_DROP_CACHE_AFTER_LOAD must be 0 or 1" >&2; exit 2;; esac
command -v setsid >/dev/null 2>&1 || { echo "setsid is required for reliable SGLang process-group cleanup" >&2; exit 2; }

IFS=',' read -r -a TASK_GPU_IDS <<<"${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  TASK_GPU_IDS=()
  for ((TASK_I=0; TASK_I<TASK_GPU_COUNT; TASK_I++)); do TASK_GPU_IDS+=("${TASK_I}"); done
fi
if [[ "${#TASK_GPU_IDS[@]}" -ne "${TASK_GPU_COUNT}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must expose exactly EVAL_GPUS=${TASK_GPU_COUNT} GPUs" >&2
  exit 2
fi
TASK_GPU_ID_CSV="$(IFS=,; echo "${TASK_GPU_IDS[*]}")"

mkdir -p -- "${TASK_OUTPUT}"
export PYTHONPATH="${TASK_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

"${TASK_PYTHON}" - "${ALFWORLD_EVAL_CHECKPOINTS_TSV}" "${TASK_GPU_COUNT}" <<'PY'
import csv, pathlib, sys
path = pathlib.Path(sys.argv[1])
gpu_count = int(sys.argv[2])
rows = [
    row for row in csv.reader(path.open(encoding="utf-8"), delimiter="\t")
    if row and any(value.strip() for value in row) and not row[0].lstrip().startswith("#")
]
if not rows:
    raise SystemExit("Checkpoint TSV is empty")
seen = set()
for line, row in enumerate(rows, 1):
    if len(row) != 3 or not all(row):
        raise SystemExit(f"{path}:{line}: expected method<TAB>seed<TAB>local_hf_path")
    key = tuple(row[:2])
    if key in seen:
        raise SystemExit(f"{path}:{line}: duplicate method/seed: {key}")
    seen.add(key)
    checkpoint = pathlib.Path(row[2]).expanduser()
    if not checkpoint.is_dir():
        raise SystemExit(f"{path}:{line}: local HF checkpoint is not a directory: {checkpoint}")
if len(rows) % gpu_count:
    raise SystemExit(
        f"Checkpoint count {len(rows)} must be divisible by EVAL_GPUS={gpu_count}; "
        "a partial final wave would leave allocated GPUs idle"
    )
print(f"Validated {len(rows)} local HF checkpoints in {len(rows) // gpu_count} wave(s)")
PY

TASK_CHILD_PIDS=()
TASK_SERVER_PIDS=()
cleanup_children() {
  # Evaluators explicitly reap their ALFWorld children.  These signals are a
  # final safety net for a failed evaluator or utilization monitor.
  for TASK_PID in "${TASK_CHILD_PIDS[@]:-}"; do
    [[ -n "${TASK_PID}" ]] && kill "${TASK_PID}" 2>/dev/null || true
  done
  for TASK_PID in "${TASK_CHILD_PIDS[@]:-}"; do
    [[ -n "${TASK_PID}" ]] && wait "${TASK_PID}" 2>/dev/null || true
  done
  # SGLang launches scheduler/tokenizer worker children.  Killing only the
  # launch_server PID can leave those workers and their host/GPU allocations
  # alive while the next checkpoint starts.  Every server is therefore its
  # own session/process group and is terminated as a unit.
  for TASK_PID in "${TASK_SERVER_PIDS[@]:-}"; do
    [[ -n "${TASK_PID}" ]] && kill -TERM -- "-${TASK_PID}" 2>/dev/null || true
  done
  TASK_DEADLINE=$((SECONDS + TASK_CLEANUP_TIMEOUT))
  while (( SECONDS < TASK_DEADLINE )); do
    TASK_ALIVE=0
    for TASK_PID in "${TASK_SERVER_PIDS[@]:-}"; do
      if [[ -n "${TASK_PID}" ]] && kill -0 -- "-${TASK_PID}" 2>/dev/null; then
        TASK_ALIVE=1
      fi
    done
    (( TASK_ALIVE == 0 )) && break
    sleep 1
  done
  for TASK_PID in "${TASK_SERVER_PIDS[@]:-}"; do
    if [[ -n "${TASK_PID}" ]] && kill -0 -- "-${TASK_PID}" 2>/dev/null; then
      echo "SGLang process group ${TASK_PID} did not exit after ${TASK_CLEANUP_TIMEOUT}s; sending SIGKILL" >&2
      kill -KILL -- "-${TASK_PID}" 2>/dev/null || true
    fi
  done
  for TASK_PID in "${TASK_SERVER_PIDS[@]:-}"; do
    [[ -n "${TASK_PID}" ]] && wait "${TASK_PID}" 2>/dev/null || true
  done
  TASK_CHILD_PIDS=()
  TASK_SERVER_PIDS=()
}
trap 'TASK_STATUS=$?; cleanup_children; exit ${TASK_STATUS}' EXIT
trap 'exit 130' INT TERM

wait_for_server() {
  "${TASK_PYTHON}" - "$1" "$2" "${TASK_STARTUP_TIMEOUT}" <<'PY'
import os, sys, time, urllib.request
url, pid, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try: os.kill(pid, 0)
    except OSError: raise SystemExit(f"SGLang process {pid} exited")
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as response:
            if 200 <= response.status < 300: raise SystemExit(0)
    except OSError: pass
    time.sleep(2)
raise SystemExit(f"Timed out waiting for {url}")
PY
}

print_memory_diagnostics() {
  echo "Host memory:"
  free -h 2>/dev/null || true
  for TASK_MEMORY_FILE in \
    /sys/fs/cgroup/memory.current \
    /sys/fs/cgroup/memory.peak \
    /sys/fs/cgroup/memory.max \
    /sys/fs/cgroup/memory.events; do
    if [[ -r "${TASK_MEMORY_FILE}" ]]; then
      echo "===== ${TASK_MEMORY_FILE} ====="
      cat "${TASK_MEMORY_FILE}" || true
    fi
  done
}

condition_args() {
  case "$1" in
    matched) echo "--action-drop 0.15 --observation-loss 0.10 --burst-probability 0";;
    clean) echo "--clean";;
    action-only) echo "--action-drop 0.15 --observation-loss 0 --burst-probability 0";;
    observation-only) echo "--action-drop 0 --observation-loss 0.10 --burst-probability 0";;
    strong) echo "--action-drop 0.25 --observation-loss 0.20 --burst-probability 0";;
    burst) echo "--action-drop 0.15 --observation-loss 0.10 --burst-probability 0.10 --burst-length 3";;
    *) echo "Unknown EVAL_CONDITIONS entry: $1" >&2; return 2;;
  esac
}

measure_utilization() {
  local TASK_CSV=$1 TASK_REPORT=$2
  "${TASK_PYTHON}" - "${TASK_CSV}" "${TASK_REPORT}" "${TASK_UTIL_MIN}" <<'PY'
import csv, json, pathlib, statistics, sys
source, output, threshold = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), float(sys.argv[3])
values = {}
with source.open(encoding="utf-8") as stream:
    for row in csv.reader(stream):
        if len(row) < 3: continue
        try: values.setdefault(row[1].strip(), []).append(float(row[2].strip()))
        except ValueError: continue
if not values: raise SystemExit("No valid nvidia-smi utilization samples were collected")
per_gpu = {gpu: {"samples": len(xs), "mean": statistics.fmean(xs), "p10": sorted(xs)[max(0, int(.1*len(xs))-1)]} for gpu, xs in values.items()}
mean = statistics.fmean(item["mean"] for item in per_gpu.values())
report = {"mean_gpu_utilization": mean, "threshold": threshold, "passed": mean >= threshold, "per_gpu": per_gpu}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, sort_keys=True))
raise SystemExit(0 if report["passed"] else 3)
PY
}

TASK_METHODS=()
TASK_SEEDS=()
TASK_CHECKPOINTS=()
while IFS=$'\t' read -r TASK_METHOD TASK_SEED_ID TASK_CHECKPOINT TASK_EXTRA; do
  [[ -z "${TASK_METHOD// /}" ]] && continue
  [[ "${TASK_METHOD}" == \#* ]] && continue
  TASK_METHODS+=("${TASK_METHOD}")
  TASK_SEEDS+=("${TASK_SEED_ID}")
  TASK_CHECKPOINTS+=("${TASK_CHECKPOINT}")
done <"${ALFWORLD_EVAL_CHECKPOINTS_TSV}"

IFS=',' read -r -a TASK_CONDITION_LIST <<<"${TASK_CONDITIONS}"
TASK_CHECKPOINT_COUNT="${#TASK_CHECKPOINTS[@]}"
TASK_WAVE_COUNT=$((TASK_CHECKPOINT_COUNT / TASK_GPU_COUNT))

for ((TASK_WAVE=0; TASK_WAVE<TASK_WAVE_COUNT; TASK_WAVE++)); do
  TASK_WAVE_DIR="${TASK_OUTPUT}/_waves/wave${TASK_WAVE}"
  mkdir -p -- "${TASK_WAVE_DIR}"
  TASK_WAVE_LABELS=()
  TASK_WAVE_CHECKPOINTS=()
  TASK_WAVE_DIRS=()
  echo "Starting checkpoint wave $((TASK_WAVE + 1))/${TASK_WAVE_COUNT}; TP=1 per GPU"
  print_memory_diagnostics

  TASK_LOADER_ARGS=()
  [[ "${TASK_DROP_CACHE}" == 1 ]] && TASK_LOADER_ARGS+=(--weight-loader-drop-cache-after-load)
  for ((TASK_I=0; TASK_I<TASK_GPU_COUNT; TASK_I++)); do
    TASK_INDEX=$((TASK_WAVE * TASK_GPU_COUNT + TASK_I))
    TASK_LABEL="${TASK_METHODS[TASK_INDEX]}-s${TASK_SEEDS[TASK_INDEX]}"
    TASK_CHECKPOINT="${TASK_CHECKPOINTS[TASK_INDEX]}"
    TASK_CHECKPOINT_DIR="${TASK_OUTPUT}/${TASK_LABEL}"
    TASK_WAVE_LABELS+=("${TASK_LABEL}")
    TASK_WAVE_CHECKPOINTS+=("${TASK_CHECKPOINT}")
    TASK_WAVE_DIRS+=("${TASK_CHECKPOINT_DIR}")
    mkdir -p -- "${TASK_CHECKPOINT_DIR}/logs"
    echo "GPU ${TASK_GPU_IDS[TASK_I]}: ${TASK_LABEL} -> ${TASK_CHECKPOINT}"
    TASK_PORT=$((TASK_PORT_BASE + TASK_I))
    setsid env CUDA_VISIBLE_DEVICES="${TASK_GPU_IDS[TASK_I]}" \
      "${TASK_SGLANG_PYTHON}" -m sglang.launch_server \
      --model-path "${TASK_CHECKPOINT}" --host "${TASK_HOST}" --port "${TASK_PORT}" \
      --tp-size 1 --dtype bfloat16 --context-length "${TASK_CONTEXT}" \
      --mem-fraction-static "${TASK_MEM_FRACTION}" "${TASK_LOADER_ARGS[@]}" \
      >"${TASK_CHECKPOINT_DIR}/logs/sglang-gpu${TASK_I}.log" 2>&1 &
    TASK_SERVER_PIDS+=("$!")
    if (( TASK_I + 1 < TASK_GPU_COUNT && TASK_STARTUP_STAGGER > 0 )); then
      sleep "${TASK_STARTUP_STAGGER}"
    fi
  done

  for ((TASK_I=0; TASK_I<TASK_GPU_COUNT; TASK_I++)); do
    if ! wait_for_server "http://${TASK_HOST}:$((TASK_PORT_BASE + TASK_I))" "${TASK_SERVER_PIDS[TASK_I]}"; then
      echo "SGLang GPU ${TASK_I} (${TASK_WAVE_LABELS[TASK_I]}) failed during startup" >&2
      print_memory_diagnostics >&2
      tail -n 120 "${TASK_WAVE_DIRS[TASK_I]}/logs/sglang-gpu${TASK_I}.log" >&2 || true
      exit 1
    fi
  done

  for TASK_CONDITION in "${TASK_CONDITION_LIST[@]}"; do
    TASK_CONDITION="${TASK_CONDITION// /}"
    read -r -a TASK_NOISE_ARGS <<<"$(condition_args "${TASK_CONDITION}")"
    TASK_WAVE_CONDITION_DIR="${TASK_WAVE_DIR}/${TASK_CONDITION}"
    mkdir -p -- "${TASK_WAVE_CONDITION_DIR}"
    TASK_GPU_CSV="${TASK_WAVE_CONDITION_DIR}/gpu-utilization.csv"
    nvidia-smi -i "${TASK_GPU_ID_CSV}" --query-gpu=timestamp,index,utilization.gpu \
      --format=csv,noheader,nounits -l "${TASK_UTIL_INTERVAL}" >"${TASK_GPU_CSV}" 2>&1 &
    TASK_MONITOR_PID=$!; TASK_CHILD_PIDS+=("${TASK_MONITOR_PID}")

    TASK_EVAL_PIDS=()
    for ((TASK_I=0; TASK_I<TASK_GPU_COUNT; TASK_I++)); do
      TASK_CONDITION_DIR="${TASK_WAVE_DIRS[TASK_I]}/${TASK_CONDITION}"
      mkdir -p -- "${TASK_CONDITION_DIR}"
      TASK_LIMIT_ARGS=(); (( TASK_LIMIT > 0 )) && TASK_LIMIT_ARGS=(--limit "${TASK_LIMIT}")
      echo "Evaluating ${TASK_WAVE_LABELS[TASK_I]}/${TASK_CONDITION} on GPU ${TASK_GPU_IDS[TASK_I]}"
      "${TASK_PYTHON}" -m noise_rl.cli evaluate \
        --config "${TASK_CONFIG}" --data "${ALFWORLD_EVAL_DATA}" \
        --model "${TASK_WAVE_CHECKPOINTS[TASK_I]}" \
        --url "http://${TASK_HOST}:$((TASK_PORT_BASE + TASK_I))" \
        --repeats "${TASK_REPEATS}" --seed "${TASK_SEED}" --temperature "${TASK_TEMPERATURE}" \
        --concurrency "${TASK_CONCURRENCY}" --environment-processes "${TASK_ENV_PROCESSES}" \
        "${TASK_LIMIT_ARGS[@]}" "${TASK_NOISE_ARGS[@]}" \
        --output "${TASK_CONDITION_DIR}/episodes.jsonl" \
        >"${TASK_CONDITION_DIR}/evaluation.log" 2>&1 &
      TASK_EVAL_PIDS+=("$!"); TASK_CHILD_PIDS+=("$!")
    done

    TASK_FAILED=0
    set +e
    for ((TASK_I=0; TASK_I<TASK_GPU_COUNT; TASK_I++)); do
      wait "${TASK_EVAL_PIDS[TASK_I]}" || TASK_FAILED=1
    done
    set -e
    kill "${TASK_MONITOR_PID}" 2>/dev/null || true
    wait "${TASK_MONITOR_PID}" 2>/dev/null || true
    TASK_CHILD_PIDS=()
    (( TASK_FAILED == 0 )) || { echo "Evaluation failed; inspect */${TASK_CONDITION}/evaluation.log" >&2; exit 1; }

    set +e
    measure_utilization "${TASK_GPU_CSV}" "${TASK_WAVE_CONDITION_DIR}/gpu-utilization.json"
    TASK_UTIL_STATUS=$?
    set -e
    for ((TASK_I=0; TASK_I<TASK_GPU_COUNT; TASK_I++)); do
      cp "${TASK_WAVE_CONDITION_DIR}/gpu-utilization.json" \
        "${TASK_WAVE_DIRS[TASK_I]}/${TASK_CONDITION}/gpu-utilization.json"
    done
    if (( TASK_UTIL_STATUS != 0 )) && [[ "${TASK_ALLOW_LOW_UTIL}" != 1 ]]; then
      echo "GPU utilization gate failed for wave ${TASK_WAVE}/${TASK_CONDITION}" >&2
      exit 3
    fi
  done

  cleanup_children
  if (( TASK_CLEANUP_PAUSE > 0 )); then
    echo "Waiting ${TASK_CLEANUP_PAUSE}s for checkpoint resources to settle"
    sleep "${TASK_CLEANUP_PAUSE}"
  fi
done

"${TASK_PYTHON}" -m noise_rl.eval_matrix table --root "${TASK_OUTPUT}" \
  --output-tsv "${TASK_OUTPUT}/summary.tsv" --output-json "${TASK_OUTPUT}/summary.json"
echo "Evaluation matrix completed: ${TASK_OUTPUT}"
