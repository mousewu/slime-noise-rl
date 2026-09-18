#!/usr/bin/env bash
set -euo pipefail

# Run task-level incident replay and build a new, task-filtered AWM manifest.
# It intentionally leaves the input manifest untouched.
TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_TRAINER_PYTHON="${PYTHON_BIN:-python}"

: "${AWM_MANIFEST:?Set AWM_MANIFEST to the current training manifest}"
: "${AWM_INCIDENTS:?Set AWM_INCIDENTS to reviewed incidents or a diagnostics directory}"
: "${AWM_URL:?Set AWM_URL to an already-running local AWM server URL}"
TASK_REPLAY_OUTPUT="${AWM_INCIDENT_REPLAY_OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-incident-replay.jsonl}"
TASK_OUTPUT="${AWM_PREFLIGHT_MANIFEST:-${TASK_PROJECT_DIR}/data/awm/train.task-preflight.jsonl}"
TASK_REPORT="${AWM_PREFLIGHT_REPORT:-${TASK_OUTPUT}.report.json}"

for TASK_PATH in "${AWM_MANIFEST}" "${AWM_INCIDENTS}"; do
  if [[ ! -e "${TASK_PATH}" ]]; then
    echo "Required input does not exist: ${TASK_PATH}" >&2
    exit 2
  fi
done
if [[ -e "${TASK_REPLAY_OUTPUT}" || -e "${TASK_REPLAY_OUTPUT}.summary.json" ]]; then
  echo "Refusing to overwrite AWM incident replay output: ${TASK_REPLAY_OUTPUT}" >&2
  exit 2
fi
if [[ -e "${TASK_OUTPUT}" || -e "${TASK_REPORT}" ]]; then
  echo "Refusing to overwrite task-preflight manifest or report" >&2
  exit 2
fi

AWM_INCIDENT_REPLAY_OUTPUT="${TASK_REPLAY_OUTPUT}" \
  "${TASK_PROJECT_DIR}/scripts/replay_awm_tool_incidents.sh"

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${TASK_TRAINER_PYTHON}" -m noise_rl.awm_filter \
  --manifest "${AWM_MANIFEST}" \
  --audit "${TASK_REPLAY_OUTPUT}" \
  --output "${TASK_OUTPUT}" \
  --report "${TASK_REPORT}"
