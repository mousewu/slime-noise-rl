#!/usr/bin/env bash
set -euo pipefail

# Build only replay-verified planner trajectories from an existing local
# ALFWorld installation.  This script never downloads or regenerates games.
TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"
: "${ALFWORLD_ROOT:?Set ALFWORLD_ROOT to the local ALFWorld json_2.1.1 directory}"

TASK_ALFWORLD_ROOT="${ALFWORLD_ROOT}"
TASK_SFT_DATA="${SFT_DATA:-${TASK_PROJECT_DIR}/data/alfworld/train-planner-sft.jsonl}"
TASK_SFT_LIMIT="${SFT_LIMIT:-}"
TASK_MAX_ACTIONS="${SFT_MAX_ACTIONS:-40}"
TASK_PLANNER_FALLBACK="${SFT_PLANNER_FALLBACK:-0}"
TASK_STRICT="${SFT_STRICT:-0}"

case "${TASK_PLANNER_FALLBACK}" in 0|1) ;; *) echo "SFT_PLANNER_FALLBACK must be 0 or 1" >&2; exit 2 ;; esac
case "${TASK_STRICT}" in 0|1) ;; *) echo "SFT_STRICT must be 0 or 1" >&2; exit 2 ;; esac
case "${TASK_MAX_ACTIONS}" in ''|*[!0-9]*) echo "SFT_MAX_ACTIONS must be a positive integer" >&2; exit 2 ;; esac
if (( 10#${TASK_MAX_ACTIONS} < 1 )); then
  echo "SFT_MAX_ACTIONS must be a positive integer" >&2
  exit 2
fi
if [[ -n "${TASK_SFT_LIMIT}" ]] && [[ "${TASK_SFT_LIMIT}" == *[!0-9]* || "${TASK_SFT_LIMIT}" == 0 ]]; then
  echo "SFT_LIMIT must be a positive integer when set" >&2
  exit 2
fi
if [[ ! -d "${TASK_ALFWORLD_ROOT}/train" ]]; then
  echo "ALFWORLD_ROOT must contain the local train split: ${TASK_ALFWORLD_ROOT}" >&2
  exit 2
fi
if [[ -e "${TASK_SFT_DATA}" || -e "${TASK_SFT_DATA}.report.json" ]]; then
  echo "SFT data or report already exists; choose a new SFT_DATA path: ${TASK_SFT_DATA}" >&2
  exit 2
fi

TASK_ALFWORLD_ROOT="$(cd -- "${TASK_ALFWORLD_ROOT}" && pwd)"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$(dirname -- "${TASK_ALFWORLD_ROOT}")}"
export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

TASK_ARGS=(--root "${TASK_ALFWORLD_ROOT}" --output "${TASK_SFT_DATA}" --max-actions "${TASK_MAX_ACTIONS}")
if [[ -n "${TASK_SFT_LIMIT}" ]]; then
  TASK_ARGS+=(--limit "${TASK_SFT_LIMIT}")
fi
if [[ "${TASK_PLANNER_FALLBACK}" == 1 ]]; then
  TASK_ARGS+=(--planner-fallback)
fi
if [[ "${TASK_STRICT}" == 1 ]]; then
  TASK_ARGS+=(--strict)
fi

exec "${TASK_PYTHON_BIN}" -m noise_rl.sft_data "${TASK_ARGS[@]}"
