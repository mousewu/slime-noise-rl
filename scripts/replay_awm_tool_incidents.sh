#!/usr/bin/env bash
set -euo pipefail

# Replay observed AWM tool failures against a freshly reset database.  This
# script must use the isolated AWM/OpenEnv interpreter, never the Megatron
# trainer interpreter.
TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${AWM_SERVER_PYTHON_BIN:-${PYTHON_BIN:-python}}"

: "${AWM_INCIDENTS:?Set AWM_INCIDENTS to awm-diagnostic JSON/JSONL or a diagnostics directory}"
: "${AWM_URL:?Set AWM_URL to an already-running local AWM server URL}"
TASK_OUTPUT="${AWM_INCIDENT_REPLAY_OUTPUT:-${TASK_PROJECT_DIR}/runs/awm-incident-replay.jsonl}"
TASK_WORKERS="${AWM_INCIDENT_REPLAY_WORKERS:-2}"
TASK_CONNECT_TIMEOUT="${AWM_INCIDENT_REPLAY_CONNECT_TIMEOUT:-15}"
TASK_MESSAGE_TIMEOUT="${AWM_INCIDENT_REPLAY_MESSAGE_TIMEOUT:-120}"

if [[ ! -e "${AWM_INCIDENTS}" ]]; then
  echo "AWM incident source does not exist: ${AWM_INCIDENTS}" >&2
  exit 2
fi
if [[ -e "${TASK_OUTPUT}" || -e "${TASK_OUTPUT}.summary.json" ]]; then
  echo "Refusing to overwrite AWM incident replay output: ${TASK_OUTPUT}" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_incident_replay \
  --incidents "${AWM_INCIDENTS}" \
  --url "${AWM_URL}" \
  --output "${TASK_OUTPUT}" \
  --workers "${TASK_WORKERS}" \
  --connect-timeout "${TASK_CONNECT_TIMEOUT}" \
  --message-timeout "${TASK_MESSAGE_TIMEOUT}"
