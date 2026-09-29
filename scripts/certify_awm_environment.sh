#!/usr/bin/env bash
set -euo pipefail

# Certify a frozen AWM training subset from two independent evidence sources:
#   1. successful teacher replay rows provide exact action witnesses;
#   2. clean student replay rows estimate operational stability/learnability.
# The AWM service must already be running.  This script starts no model,
# SGLang, Ray, Slime, or training job and performs no downloads.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"

: "${AWM_MANIFEST:?Set AWM_MANIFEST to the train-only candidate manifest}"
: "${AWM_WITNESS_REPLAY:?Set AWM_WITNESS_REPLAY to clean teacher replay JSONL}"
: "${AWM_POLICY_REPLAY:?Set AWM_POLICY_REPLAY to clean student replay JSONL}"
: "${AWM_URL:?Set AWM_URL to the already-running local AWM server}"

TASK_ENV_OUTPUT="${AWM_CERTIFIED_ENV_MANIFEST:-${TASK_PROJECT_DIR}/data/awm/train.environment-certified.jsonl}"
TASK_RL_OUTPUT="${AWM_CERTIFIED_RL_MANIFEST:-${TASK_PROJECT_DIR}/data/awm/train.rl-certified.jsonl}"
TASK_QUARANTINE="${AWM_CERTIFICATION_QUARANTINE:-${TASK_PROJECT_DIR}/runs/awm-certification.quarantine.jsonl}"
TASK_EVIDENCE="${AWM_CERTIFICATION_EVIDENCE:-${TASK_PROJECT_DIR}/runs/awm-certification.evidence.jsonl}"
TASK_REPORT="${AWM_CERTIFICATION_REPORT:-${TASK_PROJECT_DIR}/runs/awm-certification.report.json}"

for TASK_PATH in "${AWM_MANIFEST}" "${AWM_WITNESS_REPLAY}" "${AWM_POLICY_REPLAY}"; do
  [[ -f "${TASK_PATH}" ]] || { echo "Required local file does not exist: ${TASK_PATH}" >&2; exit 2; }
done

TASK_ARGS=(
  --manifest "${AWM_MANIFEST}"
  --witness-replay "${AWM_WITNESS_REPLAY}"
  --policy-replay "${AWM_POLICY_REPLAY}"
  --url "${AWM_URL}"
  --environment-output "${TASK_ENV_OUTPUT}"
  --rl-output "${TASK_RL_OUTPUT}"
  --quarantine-output "${TASK_QUARANTINE}"
  --evidence-output "${TASK_EVIDENCE}"
  --report-output "${TASK_REPORT}"
  --workers "${AWM_CERTIFICATION_WORKERS:-4}"
  --timeout "${AWM_CERTIFICATION_TIMEOUT:-180}"
  --reset-repeats "${AWM_CERTIFICATION_RESET_REPEATS:-3}"
  --witness-repeats "${AWM_CERTIFICATION_WITNESS_REPEATS:-3}"
  --counterfactual-candidates "${AWM_CERTIFICATION_COUNTERFACTUAL_CANDIDATES:-3}"
  --policy-min-attempts "${AWM_CERTIFICATION_POLICY_MIN_ATTEMPTS:-16}"
  --policy-min-success-rate "${AWM_CERTIFICATION_POLICY_MIN_SUCCESS_RATE:-0.125}"
  --policy-max-success-rate "${AWM_CERTIFICATION_POLICY_MAX_SUCCESS_RATE:-0.875}"
  --policy-max-environment-errors "${AWM_CERTIFICATION_POLICY_MAX_ENVIRONMENT_ERRORS:-0}"
  --policy-max-format-errors "${AWM_CERTIFICATION_POLICY_MAX_FORMAT_ERRORS:-0}"
)

if [[ "${AWM_CERTIFICATION_REQUIRE_DETERMINISTIC_OBSERVATIONS:-0}" == "1" ]]; then
  TASK_ARGS+=(--require-deterministic-observations)
fi
if [[ -n "${AWM_CERTIFICATION_SERVER_REVISION:-}" ]]; then
  TASK_ARGS+=(--server-revision "${AWM_CERTIFICATION_SERVER_REVISION}")
fi
if [[ "${AWM_CERTIFICATION_RESUME:-0}" == "1" ]]; then
  TASK_ARGS+=(--resume)
else
  for TASK_PATH in \
    "${TASK_ENV_OUTPUT}" "${TASK_RL_OUTPUT}" "${TASK_QUARANTINE}" \
    "${TASK_EVIDENCE}" "${TASK_REPORT}"; do
    [[ ! -e "${TASK_PATH}" ]] || { echo "Refusing to overwrite: ${TASK_PATH}" >&2; exit 2; }
  done
fi

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo "Starting AWM environment certification; no model inference or training will be started."
echo "Candidate manifest: ${AWM_MANIFEST}"
echo "Teacher witness replay: ${AWM_WITNESS_REPLAY}"
echo "Student policy replay: ${AWM_POLICY_REPLAY}"
echo "Resumable evidence: ${TASK_EVIDENCE}"
exec "${TASK_PYTHON_BIN}" -m noise_rl.awm_certify "${TASK_ARGS[@]}"
