#!/usr/bin/env bash
set -euo pipefail

# Remove whole scenarios with confirmed bad tools.  Removing only a tool from
# a schema would keep tasks whose goal may require that tool and poison RL
# rewards with structurally impossible episodes.
TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_MANIFEST:?Set AWM_MANIFEST to the input training manifest}"
: "${AWM_TOOL_AUDIT_OUTPUT:?Set AWM_TOOL_AUDIT_OUTPUT to audit_awm_tools.sh output}"
TASK_OUTPUT="${AWM_FILTERED_MANIFEST:-${TASK_PROJECT_DIR}/data/awm/train.tool-filtered.jsonl}"
TASK_REPORT="${AWM_FILTER_REPORT:-${TASK_OUTPUT}.report.json}"

for TASK_PATH in "${AWM_MANIFEST}" "${AWM_TOOL_AUDIT_OUTPUT}"; do
  if [[ ! -f "${TASK_PATH}" ]]; then
    echo "Required file does not exist: ${TASK_PATH}" >&2
    exit 2
  fi
done
if [[ -e "${TASK_OUTPUT}" || -e "${TASK_REPORT}" ]]; then
  echo "Refusing to overwrite filtered manifest or report" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_filter \
  --manifest "${AWM_MANIFEST}" \
  --audit "${AWM_TOOL_AUDIT_OUTPUT}" \
  --output "${TASK_OUTPUT}" \
  --report "${TASK_REPORT}"
