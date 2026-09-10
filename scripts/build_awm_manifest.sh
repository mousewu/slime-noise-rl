#!/usr/bin/env bash
set -euo pipefail

# Build train and scenario-disjoint valid_unseen manifests from existing local
# AgentWorldModel-1K assets.  This script never downloads data or imports the
# OpenEnv server, so it is safe to run on an air-gapped training host.
TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_DATA_DIR:?Set AWM_DATA_DIR to the local AgentWorldModel-1K directory}"
TASK_TRAIN_OUTPUT="${AWM_TRAIN_MANIFEST:-${TASK_PROJECT_DIR}/data/awm/train.jsonl}"
TASK_VALID_OUTPUT="${AWM_VALID_UNSEEN_MANIFEST:-${TASK_PROJECT_DIR}/data/awm/valid_unseen.jsonl}"
TASK_SPLIT_SEED="${AWM_SPLIT_SEED:-20260910}"
TASK_VALID_FRACTION="${AWM_VALID_SCENARIO_FRACTION:-0.2}"
TASK_REPORT="${AWM_MANIFEST_REPORT:-${TASK_PROJECT_DIR}/data/awm/split-report.json}"
TASK_READ_ONLY_TOOLS="${AWM_READ_ONLY_TOOLS:-}"

if [[ ! -d "${AWM_DATA_DIR}" ]]; then
  echo "AWM_DATA_DIR must be an existing local directory: ${AWM_DATA_DIR}" >&2
  exit 2
fi
for TASK_FILE in \
  gen_scenario.jsonl gen_tasks.jsonl gen_db.jsonl gen_sample.jsonl \
  gen_envs.jsonl gen_verifier.jsonl gen_verifier.pure_code.jsonl; do
  if [[ ! -f "${AWM_DATA_DIR}/${TASK_FILE}" ]]; then
    echo "Missing local AWM data file: ${AWM_DATA_DIR}/${TASK_FILE}" >&2
    exit 2
  fi
done
if [[ -n "${TASK_READ_ONLY_TOOLS}" && ! -f "${TASK_READ_ONLY_TOOLS}" ]]; then
  echo "AWM_READ_ONLY_TOOLS must name an existing JSON file: ${TASK_READ_ONLY_TOOLS}" >&2
  exit 2
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

TASK_ARGS=(
  --data-root "${AWM_DATA_DIR}"
  --train-output "${TASK_TRAIN_OUTPUT}"
  --valid-unseen-output "${TASK_VALID_OUTPUT}"
  --valid-scenario-fraction "${TASK_VALID_FRACTION}"
  --seed "${TASK_SPLIT_SEED}"
  --report "${TASK_REPORT}"
)
if [[ -n "${TASK_READ_ONLY_TOOLS}" ]]; then
  TASK_ARGS+=(--read-only-tools "${TASK_READ_ONLY_TOOLS}")
fi

exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_manifest "${TASK_ARGS[@]}"
