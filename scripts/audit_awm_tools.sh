#!/usr/bin/env bash
set -euo pipefail

# Confirm generated-AWM route failures by actually calling only the tools whose
# FastAPI routes are statically shadowed.  Use the isolated AWM interpreter:
# the Slime/Megatron environment intentionally does not install OpenEnv.
TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${AWM_SERVER_PYTHON_BIN:-${PYTHON_BIN:-python}}"

: "${AWM_DATA_DIR:?Set AWM_DATA_DIR to local AgentWorldModel-1K data}"
: "${AWM_URL:?Set AWM_URL to an already-running local AWM server URL}"
TASK_OUTPUT="${AWM_TOOL_AUDIT_OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-tool-audit.jsonl}"
TASK_WORKERS="${AWM_TOOL_AUDIT_WORKERS:-2}"
TASK_MAX_TARGETS="${AWM_TOOL_AUDIT_MAX_TARGETS:-}"
TASK_SCHEMA_ALL="${AWM_TOOL_AUDIT_SCHEMA_ALL:-1}"

if [[ ! -f "${AWM_DATA_DIR}/gen_envs.jsonl" ]]; then
  echo "Missing AWM source data: ${AWM_DATA_DIR}/gen_envs.jsonl" >&2
  exit 2
fi
if [[ -e "${TASK_OUTPUT}" || -e "${TASK_OUTPUT}.summary.json" ]]; then
  echo "Refusing to overwrite existing AWM tool audit: ${TASK_OUTPUT}" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
TASK_ARGS=(--data-dir "${AWM_DATA_DIR}" --url "${AWM_URL}" --output "${TASK_OUTPUT}" --workers "${TASK_WORKERS}")
if [[ -n "${TASK_MAX_TARGETS}" ]]; then
  TASK_ARGS+=(--max-targets "${TASK_MAX_TARGETS}")
fi
case "${TASK_SCHEMA_ALL}" in
  0) ;;
  1) TASK_ARGS+=(--schema-all) ;;
  *) echo "AWM_TOOL_AUDIT_SCHEMA_ALL must be 0 or 1" >&2; exit 2 ;;
esac
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_tool_audit "${TASK_ARGS[@]}"
