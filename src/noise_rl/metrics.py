import math
import random
from collections import Counter, defaultdict
from statistics import mean

from .agent import Trajectory
from .config import ExperimentConfig
from .sampling import SamplingPlan


def episode_record(
    trajectory: Trajectory,
    plan: SamplingPlan,
    config: ExperimentConfig,
    *,
    checkpoint=None,
):
    return {
        "schema": 2,
        "plan": plan.to_dict(),
        "config": config.to_dict(),
        "checkpoint": checkpoint,
        "success": trajectory.success,
        "termination": trajectory.termination,
        "generated_tokens": trajectory.generated_tokens,
        "inference_input_tokens": trajectory.inference_input_tokens,
        "context_tokens": len(trajectory.tokens),
        "tool_calls": trajectory.tool_calls,
        "elapsed_seconds": trajectory.elapsed_seconds,
        "model_requests": trajectory.model_requests,
        "model_request_seconds": trajectory.model_request_seconds,
        "environment_queue_seconds": trajectory.environment_queue_seconds,
        "environment_step_seconds": trajectory.environment_step_seconds,
        "in_flight_episodes_at_start": trajectory.in_flight_episodes_at_start,
        "turns": len([s for s in trajectory.segments if s.trainable]),
        "format_errors": sum(step["format_error"] for step in trajectory.steps),
        "steps": trajectory.steps,
        "fault_audit": trajectory.audit,
    }


def bootstrap_mean(values, seed=20260904, resamples=2000):
    if not values or resamples < 1:
        raise ValueError("Nonempty values and positive resamples are required")
    rng = random.Random(seed)
    boot = sorted(mean(rng.choices(values, k=len(values))) for _ in range(resamples))
    return [boot[int(0.025 * (resamples - 1))], boot[int(0.975 * (resamples - 1))]]


def summarize(records: list[dict]):
    if not records:
        raise ValueError("No episode records")
    by_task = defaultdict(list)
    for record in records:
        by_task[record["plan"]["task_id"]].append(record)
    task_rates = [mean(float(r["success"]) for r in rows) for rows in by_task.values()]
    return {
        "episodes": len(records),
        "tasks": len(by_task),
        "success_rate_task_macro": mean(task_rates),
        "success_rate_episode_micro": mean(float(r["success"]) for r in records),
        "success_rate_task_bootstrap_95ci": bootstrap_mean(task_rates),
        "mean_generated_tokens": mean(r["generated_tokens"] for r in records),
        "total_generated_tokens": sum(r["generated_tokens"] for r in records),
        "total_inference_input_tokens": sum(
            r["inference_input_tokens"] for r in records
        ),
        "mean_tool_calls": mean(r["tool_calls"] for r in records),
        "terminations": dict(Counter(r["termination"] for r in records)),
        "note": "CI resamples tasks, not correlated rollouts; it does not replace multiple training seeds.",
    }


def trace_metrics(
    records: list[dict], *, prefix: str, step: int
) -> dict[str, int | float]:
    """Summarize completed trace records into finite scalar tracking metrics.

    Trace files remain the detailed source of truth.  This compact view is
    deliberately limited to scalar quantities so it can be sent safely from a
    Ray worker to one SwanLab owner process.
    """
    if not records:
        raise ValueError("No episode records")
    if not prefix or prefix.endswith("/"):
        raise ValueError(
            "prefix must be a nonempty metric namespace without a trailing slash"
        )

    count = len(records)
    faults = [event for record in records for event in record.get("fault_audit", [])]
    terminations = Counter(str(record["termination"]) for record in records)
    metrics: dict[str, int | float] = {
        f"{prefix}/step": step,
        f"{prefix}/episodes": count,
        f"{prefix}/tasks": len({record["plan"]["task_id"] for record in records}),
        f"{prefix}/success_rate": mean(float(record["success"]) for record in records),
        f"{prefix}/generated_tokens/mean": mean(
            record["generated_tokens"] for record in records
        ),
        f"{prefix}/generated_tokens/total": sum(
            record["generated_tokens"] for record in records
        ),
        f"{prefix}/inference_input_tokens/mean": mean(
            record["inference_input_tokens"] for record in records
        ),
        f"{prefix}/inference_input_tokens/total": sum(
            record["inference_input_tokens"] for record in records
        ),
        f"{prefix}/context_tokens/mean": mean(
            record["context_tokens"] for record in records
        ),
        f"{prefix}/tool_calls/mean": mean(record["tool_calls"] for record in records),
        f"{prefix}/tool_calls/total": sum(record["tool_calls"] for record in records),
        f"{prefix}/turns/mean": mean(record["turns"] for record in records),
        f"{prefix}/format_errors/total": sum(
            record["format_errors"] for record in records
        ),
        f"{prefix}/elapsed_seconds/mean": mean(
            record["elapsed_seconds"] for record in records
        ),
        f"{prefix}/elapsed_seconds/max": max(
            record["elapsed_seconds"] for record in records
        ),
        f"{prefix}/elapsed_seconds/total": sum(
            record["elapsed_seconds"] for record in records
        ),
        f"{prefix}/model_requests/mean": mean(
            record.get("model_requests", record.get("turns", 0)) for record in records
        ),
        f"{prefix}/model_requests/total": sum(
            record.get("model_requests", record.get("turns", 0)) for record in records
        ),
        f"{prefix}/model_request_seconds/mean": mean(
            record.get("model_request_seconds", 0.0) for record in records
        ),
        f"{prefix}/model_request_seconds/total": sum(
            record.get("model_request_seconds", 0.0) for record in records
        ),
        f"{prefix}/environment_step_seconds/mean": mean(
            record.get("environment_step_seconds", 0.0) for record in records
        ),
        f"{prefix}/environment_step_seconds/total": sum(
            record.get("environment_step_seconds", 0.0) for record in records
        ),
        f"{prefix}/environment_queue_seconds/mean": mean(
            record.get("environment_queue_seconds", 0.0) for record in records
        ),
        f"{prefix}/environment_queue_seconds/total": sum(
            record.get("environment_queue_seconds", 0.0) for record in records
        ),
        f"{prefix}/in_flight_episodes_at_start/mean": mean(
            record.get("in_flight_episodes_at_start", 1) for record in records
        ),
        f"{prefix}/in_flight_episodes_at_start/max": max(
            record.get("in_flight_episodes_at_start", 1) for record in records
        ),
        f"{prefix}/faults/action_drop_rate": (
            sum(bool(event.get("dropped")) for event in faults) / len(faults)
            if faults
            else 0.0
        ),
        f"{prefix}/faults/observation_loss_rate": (
            sum(bool(event.get("observation_lost")) for event in faults) / len(faults)
            if faults
            else 0.0
        ),
    }
    for termination, value in terminations.items():
        metrics[f"{prefix}/termination/{termination}_rate"] = value / count
    return metrics


def paired_comparison(left: list[dict], right: list[dict]):
    def index(rows):
        result = {}
        for row in rows:
            p = row["plan"]
            if not p["evaluation"]:
                raise ValueError("Compare held-out evaluations, not training rollouts")
            key = (p["task_id"], p["group_id"], p["rank"])
            if key in result:
                raise ValueError("Duplicate evaluation sample identity")
            result[key] = row
        return result

    a, b = index(left), index(right)
    if not a or set(a) != set(b):
        raise ValueError(
            "Paired evaluations require exactly the same tasks and repetitions"
        )
    differences = defaultdict(list)
    for key, row in a.items():
        other = b[key]
        if row["plan"] != other["plan"]:
            raise ValueError("Evaluation seeds/plans differ")
        for field in (
            "noise",
            "max_turns",
            "max_tool_calls",
            "max_context_tokens",
            "max_generated_tokens",
            "max_tokens_per_turn",
            "retry_limit",
        ):
            if row["config"][field] != other["config"][field]:
                raise ValueError(f"Evaluation conditions differ: {field}")
        differences[key[0]].append(float(other["success"]) - float(row["success"]))
    delta = [mean(values) for values in differences.values()]
    return {
        "tasks": len(delta),
        "success_delta_right_minus_left": mean(delta),
        "task_paired_bootstrap_95ci": bootstrap_mean(delta),
    }


def balanced_variance_components(matrix: list[list[float]]):
    """One-way random-effects diagnostic: environment blocks x independent policy draws.

    Between-environment variance is NOT an oracle measure of 'unlearnable noise'.
    Interactions remain in the within-block component. Needs repeated draws, not 2x4 alone.
    """
    if (
        len(matrix) < 2
        or len(matrix[0]) < 2
        or any(len(r) != len(matrix[0]) for r in matrix)
    ):
        raise ValueError("Need a balanced matrix with at least two rows and columns")
    if not all(math.isfinite(x) for row in matrix for x in row):
        raise ValueError("Non-finite outcome")
    n = len(matrix[0])
    means = [mean(row) for row in matrix]
    within = mean(sum((x - mean(row)) ** 2 for x in row) / (n - 1) for row in matrix)
    between_means = sum((m - mean(means)) ** 2 for m in means) / (len(means) - 1)
    return {
        "within_environment_variance": within,
        "between_environment_component_unclipped": between_means - within / n,
        "note": "Finite-sample ANOVA diagnostic, not causal attribution; negative estimates are retained.",
    }
