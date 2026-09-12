#!/usr/bin/env bash
set -euo pipefail

# Start AWM in its own Python environment. It must never share the Slime /
# Megatron interpreter because AWM's mcp-agent dependency requires NumPy 2.x.
# This script never downloads models or task data.

: "${AWM_SERVER_PYTHON_BIN:?Set AWM_SERVER_PYTHON_BIN to the isolated AWM-server Python executable}"
: "${OPENENV_DIR:?Set OPENENV_DIR to the local Hugging Face OpenEnv checkout}"
: "${AWM_DATA_DIR:?Set AWM_DATA_DIR to the local AgentWorldModel-1K directory}"
: "${RUNTIME_TMPDIR:?Set RUNTIME_TMPDIR to a large local disk directory}"

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${AWM_SERVER_PYTHON_BIN}"
TASK_HOST="${AWM_HOST:-127.0.0.1}"
TASK_PORT="${AWM_PORT:-8899}"
TASK_BACKGROUND="${AWM_SERVER_BACKGROUND:-0}"
TASK_INSTALL_DEPS="${AWM_SERVER_INSTALL_DEPS:-0}"
TASK_LOG="${AWM_SERVER_LOG:-${RUNTIME_TMPDIR}/awm-server-${TASK_PORT}.log}"
TASK_PID_FILE="${AWM_SERVER_PID_FILE:-${RUNTIME_TMPDIR}/awm-server-${TASK_PORT}.pid}"
TASK_DIAGNOSTICS_DIR="${AWM_DIAGNOSTICS_DIR:-${RUNTIME_TMPDIR}/awm-session-diagnostics}"

for TASK_FLAG in "${TASK_BACKGROUND}" "${TASK_INSTALL_DEPS}"; do
  case "${TASK_FLAG}" in 0|1) ;; *) echo "Boolean options must be 0 or 1" >&2; exit 2 ;; esac
done
for TASK_DIRECTORY in "${OPENENV_DIR}" "${AWM_DATA_DIR}"; do
  if [[ ! -d "${TASK_DIRECTORY}" ]]; then
    echo "Required local directory does not exist: ${TASK_DIRECTORY}" >&2
    exit 2
  fi
done
if [[ ! -f "${OPENENV_DIR}/pyproject.toml" || ! -f "${OPENENV_DIR}/envs/agent_world_model_env/pyproject.toml" ]]; then
  echo "OPENENV_DIR must contain OpenEnv and envs/agent_world_model_env" >&2
  exit 2
fi
TASK_AWM_FILES=(
  gen_scenario.jsonl gen_tasks.jsonl gen_db.jsonl gen_sample.jsonl
  gen_envs.jsonl gen_verifier.jsonl gen_verifier.pure_code.jsonl
)
for TASK_AWM_FILE in "${TASK_AWM_FILES[@]}"; do
  if [[ ! -f "${AWM_DATA_DIR}/${TASK_AWM_FILE}" ]]; then
    echo "Missing local AWM data file: ${AWM_DATA_DIR}/${TASK_AWM_FILE}" >&2
    echo "All seven files are required because network dataset downloads are disabled." >&2
    exit 2
  fi
done

mkdir -p -- "${RUNTIME_TMPDIR}"
mkdir -p -- "${TASK_DIAGNOSTICS_DIR}"
export TMPDIR="$(cd -- "${RUNTIME_TMPDIR}" && pwd)"
export OPENENV_DIR="$(cd -- "${OPENENV_DIR}" && pwd)"
export AWM_DATA_DIR="$(cd -- "${AWM_DATA_DIR}" && pwd)"
export NOISE_RL_AWM_DIAGNOSTICS_DIR="$(cd -- "${TASK_DIAGNOSTICS_DIR}" && pwd)"
export PYTHONPATH="${TASK_PROJECT_DIR}/src:${OPENENV_DIR}/src:${OPENENV_DIR}/envs${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [[ "${TASK_INSTALL_DEPS}" == 1 ]]; then
  echo "Installing OpenEnv/AWM only in the isolated AWM server environment"
  "${TASK_PYTHON_BIN}" -m pip install \
    -e "${OPENENV_DIR}" \
    -e "${OPENENV_DIR}/envs/agent_world_model_env"
fi

"${TASK_PYTHON_BIN}" - <<'PY'
import importlib
import numpy

for package in ("openenv", "agent_world_model_env", "fastapi", "uvicorn"):
    importlib.import_module(package)
if int(numpy.__version__.split(".", 1)[0]) < 2:
    raise RuntimeError(
        f"AWM server dependency graph requires NumPy 2.x, found {numpy.__version__}. "
        "Install OpenEnv/AWM in a dedicated server environment."
    )
print(f"AWM server Python: {__import__('sys').executable}; NumPy: {numpy.__version__}")
PY

TASK_COMMAND=("${TASK_PYTHON_BIN}" -m noise_rl.awm_server_entry --host "${TASK_HOST}" --port "${TASK_PORT}")
if [[ "${TASK_BACKGROUND}" == 1 ]]; then
  if [[ -e "${TASK_PID_FILE}" ]]; then
    echo "AWM_SERVER_PID_FILE already exists: ${TASK_PID_FILE}" >&2
    exit 2
  fi
  echo "Starting AWM server at http://${TASK_HOST}:${TASK_PORT}; log: ${TASK_LOG}; diagnostics: ${NOISE_RL_AWM_DIAGNOSTICS_DIR}"
  ENABLE_WEB_INTERFACE=false "${TASK_COMMAND[@]}" >"${TASK_LOG}" 2>&1 &
  TASK_PID=$!
  printf '%s\n' "${TASK_PID}" >"${TASK_PID_FILE}"
  echo "AWM server PID: ${TASK_PID}"
else
  echo "Starting foreground AWM server at http://${TASK_HOST}:${TASK_PORT}; diagnostics: ${NOISE_RL_AWM_DIAGNOSTICS_DIR}"
  exec env ENABLE_WEB_INTERFACE=false "${TASK_COMMAND[@]}"
fi
