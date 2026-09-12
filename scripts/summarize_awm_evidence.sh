#!/usr/bin/env bash
set -euo pipefail

# Summarize bounded AWM verifier-evidence bundles without jq or OpenEnv.
# It only reads the diagnostics directory and refuses to overwrite a report.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_DIAGNOSTICS_DIR:?Set AWM_DIAGNOSTICS_DIR to awm-session-diagnostics}"
: "${AWM_EVIDENCE_REPORT_DIR:?Set AWM_EVIDENCE_REPORT_DIR to a new report directory}"

TASK_MAX_EXAMPLES="${AWM_EVIDENCE_REPORT_MAX_EXAMPLES:-5}"
case "${TASK_MAX_EXAMPLES}" in ''|*[!0-9]*|0) echo "AWM_EVIDENCE_REPORT_MAX_EXAMPLES must be a positive integer" >&2; exit 2 ;; esac
if [[ ! -d "${AWM_DIAGNOSTICS_DIR}" && ! -f "${AWM_DIAGNOSTICS_DIR}" ]]; then
  echo "AWM_DIAGNOSTICS_DIR does not exist: ${AWM_DIAGNOSTICS_DIR}" >&2
  exit 2
fi
if [[ -e "${AWM_EVIDENCE_REPORT_DIR}" ]]; then
  echo "AWM_EVIDENCE_REPORT_DIR already exists; choose a new directory: ${AWM_EVIDENCE_REPORT_DIR}" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_evidence_report \
  --input "${AWM_DIAGNOSTICS_DIR}" \
  --output-dir "${AWM_EVIDENCE_REPORT_DIR}" \
  --max-examples "${TASK_MAX_EXAMPLES}"
