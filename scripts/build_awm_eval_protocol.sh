#!/usr/bin/env bash
set -euo pipefail

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"
: "${AWM_ENV_CERTIFIED_MANIFEST:?Set AWM_ENV_CERTIFIED_MANIFEST}"
: "${AWM_RL_CERTIFIED_MANIFEST:?Set AWM_RL_CERTIFIED_MANIFEST}"
TASK_OUTPUT="${AWM_EVAL_PROTOCOL_DIR:-${TASK_PROJECT_DIR}/data/awm/eval-certified128}"

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_eval_protocol \
  --environment-manifest "${AWM_ENV_CERTIFIED_MANIFEST}" \
  --rl-manifest "${AWM_RL_CERTIFIED_MANIFEST}" \
  --output-dir "${TASK_OUTPUT}"
