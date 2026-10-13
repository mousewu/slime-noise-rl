#!/usr/bin/env bash
set -euo pipefail

# Add an SFT baseline to an existing rollout100/200/300 single-GPU evaluation.
# The three RL replays are reused after their protocol and sampling settings
# are verified. Only the SFT checkpoint is loaded and evaluated here.

TASK_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON_BIN="${PYTHON_BIN:-python}"
: "${AWM_URL:?Set AWM_URL to the running local AWM service}"
: "${AWM_EVAL_PROTOCOL_DIR:?Set AWM_EVAL_PROTOCOL_DIR used by the RL evaluation}"
: "${EVAL_CHECKPOINT_SFT:?Set EVAL_CHECKPOINT_SFT to the final SFT HF checkpoint directory}"
: "${AWM_EXISTING_ROLLOUT_EVAL_DIR:?Set AWM_EXISTING_ROLLOUT_EVAL_DIR to the completed rollout100/200/300 evaluation}"

TASK_PROTOCOL="${AWM_EVAL_PROTOCOL_DIR}/protocol.json"
TASK_MANIFEST="${AWM_EVAL_PROTOCOL_DIR}/replay_union.jsonl"
TASK_EXISTING_DIR="${AWM_EXISTING_ROLLOUT_EVAL_DIR}"
TASK_SFT_DIR="${TASK_EXISTING_DIR}/sft"
TASK_REPEATS="${AWM_CHECKPOINT_EVAL_REPEATS:-4}"
TASK_SEED="${AWM_CHECKPOINT_EVAL_SEED:-20261011}"
TASK_TEMPERATURE="${AWM_CHECKPOINT_EVAL_TEMPERATURE:-0.8}"
TASK_CONCURRENCY="${AWM_REPLAY_CONCURRENCY_PER_GPU:-96}"
TASK_OUTPUT_JSON="${AWM_SFT_COMPARISON_JSON:-${TASK_EXISTING_DIR}/summary-with-sft.json}"
TASK_OUTPUT_TSV="${AWM_SFT_COMPARISON_TSV:-${TASK_EXISTING_DIR}/summary-with-sft.tsv}"

[[ -f "${TASK_PROTOCOL}" && -f "${TASK_MANIFEST}" ]] || {
  echo "Evaluation protocol is incomplete: ${AWM_EVAL_PROTOCOL_DIR}" >&2; exit 2;
}
[[ -d "${EVAL_CHECKPOINT_SFT}" ]] || {
  echo "SFT HF checkpoint does not exist: ${EVAL_CHECKPOINT_SFT}" >&2; exit 2;
}
[[ -d "${TASK_EXISTING_DIR}" ]] || {
  echo "Existing rollout evaluation directory does not exist: ${TASK_EXISTING_DIR}" >&2; exit 2;
}
[[ ! -e "${TASK_SFT_DIR}" ]] || {
  echo "Refusing to overwrite existing SFT evaluation: ${TASK_SFT_DIR}" >&2; exit 2;
}
[[ ! -e "${TASK_OUTPUT_JSON}" && ! -e "${TASK_OUTPUT_TSV}" ]] || {
  echo "Refusing to overwrite an existing SFT comparison summary" >&2; exit 2;
}

TASK_RL_REPLAYS=()
for TASK_LABEL in rollout100 rollout200 rollout300; do
  TASK_REPLAY="${TASK_EXISTING_DIR}/${TASK_LABEL}/replay.jsonl"
  TASK_REPLAY_SUMMARY="${TASK_REPLAY}.summary.json"
  [[ -f "${TASK_REPLAY}" && -f "${TASK_REPLAY_SUMMARY}" ]] || {
    echo "Missing completed ${TASK_LABEL} replay or summary under ${TASK_EXISTING_DIR}" >&2; exit 2;
  }
  TASK_RL_REPLAYS+=("${TASK_REPLAY}")
done

export PYTHONPATH="${TASK_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
TASK_BUDGETS="$("${TASK_PYTHON_BIN}" - \
  "${TASK_MANIFEST}" "${TASK_REPEATS}" "${TASK_SEED}" "${TASK_TEMPERATURE}" \
  "${TASK_RL_REPLAYS[@]}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

manifest = Path(sys.argv[1]).expanduser().resolve(strict=True)
expected_repeats = int(sys.argv[2])
expected_seed = int(sys.argv[3])
expected_temperature = float(sys.argv[4])
replays = [Path(value).expanduser().resolve(strict=True) for value in sys.argv[5:]]

digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
reference = None
for replay in replays:
    summary = json.loads(Path(str(replay) + ".summary.json").read_text(encoding="utf-8"))
    observed = {
        "data_sha256": summary.get("data_sha256"),
        "repeats": summary.get("repeats"),
        "seed": summary.get("seed"),
        "temperature": summary.get("policy", {}).get("temperature"),
        "max_turns": summary.get("policy", {}).get("max_turns"),
        "max_tool_calls": summary.get("policy", {}).get("max_tool_calls"),
        "max_context_tokens": summary.get("policy", {}).get("max_context_tokens"),
        "max_generated_tokens": summary.get("policy", {}).get("max_generated_tokens"),
        "max_tokens_per_turn": summary.get("policy", {}).get("max_tokens_per_turn"),
    }
    if observed["data_sha256"] != digest:
        raise SystemExit(f"{replay} used a different evaluation manifest")
    if observed["repeats"] != expected_repeats or observed["seed"] != expected_seed:
        raise SystemExit(
            f"{replay} is not paired: expected repeats={expected_repeats}, seed={expected_seed}; "
            f"observed repeats={observed['repeats']}, seed={observed['seed']}"
        )
    if not math.isclose(float(observed["temperature"]), expected_temperature):
        raise SystemExit(
            f"{replay} temperature differs: expected {expected_temperature}, "
            f"observed {observed['temperature']}"
        )
    budgets = {key: value for key, value in observed.items() if key not in {
        "data_sha256", "repeats", "seed", "temperature"
    }}
    if reference is None:
        reference = budgets
    elif budgets != reference:
        raise SystemExit(f"{replay} uses different rollout budgets: {budgets} != {reference}")
names = (
    "max_turns", "max_tool_calls", "max_context_tokens",
    "max_generated_tokens", "max_tokens_per_turn",
)
if any(type(reference.get(name)) is not int or reference[name] < 1 for name in names):
    raise SystemExit(f"Replay summary has invalid rollout budgets: {reference}")
print("\t".join(str(reference[name]) for name in names))
PY
)"
IFS=$'\t' read -r TASK_MAX_TURNS TASK_MAX_TOOL_CALLS TASK_MAX_CONTEXT_TOKENS \
  TASK_MAX_GENERATED_TOKENS TASK_MAX_TOKENS_PER_TURN <<<"${TASK_BUDGETS}"
echo "Validated paired rollout provenance; budgets: turns=${TASK_MAX_TURNS}, tools=${TASK_MAX_TOOL_CALLS}, context=${TASK_MAX_CONTEXT_TOKENS}, generated=${TASK_MAX_GENERATED_TOKENS}, per-turn=${TASK_MAX_TOKENS_PER_TURN}"

mkdir -p "${TASK_SFT_DIR}"
echo "Evaluating SFT baseline with the same manifest, seeds, budgets, and sampling settings"
env \
  HF_CHECKPOINT="${EVAL_CHECKPOINT_SFT}" \
  AWM_MANIFEST="${TASK_MANIFEST}" \
  AWM_SINGLE_REPLAY_OUTPUT="${TASK_SFT_DIR}/replay.jsonl" \
  AWM_SINGLE_REPLAY_LOG_DIR="${TASK_SFT_DIR}/logs" \
  AWM_MODEL_REPLAY_REPEATS="${TASK_REPEATS}" \
  AWM_MODEL_REPLAY_SEED="${TASK_SEED}" \
  AWM_MODEL_REPLAY_TEMPERATURE="${TASK_TEMPERATURE}" \
  AWM_REPLAY_CONCURRENCY_PER_GPU="${TASK_CONCURRENCY}" \
  MAX_TURNS="${TASK_MAX_TURNS}" \
  MAX_TOOL_CALLS="${TASK_MAX_TOOL_CALLS}" \
  MAX_CONTEXT_TOKENS="${TASK_MAX_CONTEXT_TOKENS}" \
  MAX_GENERATED_TOKENS="${TASK_MAX_GENERATED_TOKENS}" \
  MAX_TOKENS_PER_TURN="${TASK_MAX_TOKENS_PER_TURN}" \
  bash "${TASK_PROJECT_DIR}/scripts/replay_awm_1gpu.sh"

"${TASK_PYTHON_BIN}" -m noise_rl.awm_eval_summary \
  --protocol "${TASK_PROTOCOL}" \
  --checkpoint sft "${TASK_SFT_DIR}/replay.jsonl" \
  --checkpoint rollout100 "${TASK_EXISTING_DIR}/rollout100/replay.jsonl" \
  --checkpoint rollout200 "${TASK_EXISTING_DIR}/rollout200/replay.jsonl" \
  --checkpoint rollout300 "${TASK_EXISTING_DIR}/rollout300/replay.jsonl" \
  --seed "${TASK_SEED}" \
  --output-json "${TASK_OUTPUT_JSON}" \
  --output-tsv "${TASK_OUTPUT_TSV}"

echo "SFT baseline evaluation complete: ${TASK_SFT_DIR}"
echo "Four-model comparison: ${TASK_OUTPUT_TSV}"
