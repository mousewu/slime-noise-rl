#!/usr/bin/env bash
set -euo pipefail

# Turn fresh-session incident replay evidence into a compact, non-excluding
# human review queue. This script neither starts AWM/SGLang nor changes a
# training manifest.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_INCIDENT_RECHECK:?Set AWM_INCIDENT_RECHECK to recheck JSONL from replay_awm_tool_incidents.sh}"
: "${AWM_DATA_DIR:?Set AWM_DATA_DIR to the local AgentWorldModel-1K directory}"

TASK_OUTPUT="${AWM_REVIEW_QUEUE_OUTPUT:-${AWM_INCIDENT_RECHECK%.jsonl}.review-queue.jsonl}"
TASK_REPORT="${AWM_REVIEW_QUEUE_REPORT:-${TASK_OUTPUT%.jsonl}.REPORT.md}"

if [[ ! -f "${AWM_INCIDENT_RECHECK}" ]]; then
  echo "AWM_INCIDENT_RECHECK must be an existing JSONL file: ${AWM_INCIDENT_RECHECK}" >&2
  exit 2
fi
if [[ ! -f "${AWM_DATA_DIR}/gen_tasks.jsonl" ]]; then
  echo "AWM_DATA_DIR must contain gen_tasks.jsonl: ${AWM_DATA_DIR}" >&2
  exit 2
fi
if [[ -e "${TASK_OUTPUT}" || -e "${TASK_REPORT}" ]]; then
  echo "Review queue output or report already exists; choose new paths" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_incident_review_queue \
  --recheck "${AWM_INCIDENT_RECHECK}" \
  --data-root "${AWM_DATA_DIR}" \
  --output "${TASK_OUTPUT}" \
  --report "${TASK_REPORT}"
