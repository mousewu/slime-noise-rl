import math
from collections import defaultdict

from .config import ExperimentConfig
from .sampling import SamplingPlan, plan_sample


def group_advantages(rewards: list[float], plans: list[SamplingPlan], config: ExperimentConfig):
    """Return scalar sequence advantages in original sample order, before DP sharding.

    LOO without std normalization is the primary variance-reduction experiment.
    Conditional sampling does not justify claiming an unbiased clipped PPO objective.
    """
    if len(rewards) != len(plans) or not plans:
        raise ValueError("Rewards and nonempty plans must align")
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError("Non-finite reward")
    task_groups, buckets = defaultdict(list), defaultdict(list)
    for i, plan in enumerate(plans):
        if plan.evaluation:
            raise ValueError("Evaluation samples must never enter training normalization")
        expected = plan_sample(config, plan.task_id, plan.group_id, plan.rank)
        if plan != expected:
            raise ValueError("Sample plan does not match experiment configuration")
        task_groups[(plan.task_id, plan.group_id)].append(plan.rank)
        key = (plan.task_id, plan.group_id, plan.scenario_id if config.method == "matched" else -1)
        buckets[key].append(i)
    for ranks in task_groups.values():
        if sorted(ranks) != list(range(config.group_size)):
            raise ValueError("Incomplete or duplicated prompt group; do not silently normalize across tasks")
    advantages = [0.0] * len(plans)
    zero_groups = 0
    for indices in buckets.values():
        values = [rewards[i] for i in indices]
        n = len(values)
        if n < 2:
            raise ValueError("At least two samples are required for each advantage baseline")
        mean = math.fsum(values) / n
        variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
        zero_groups += variance == 0
        # Matches Slime's sample-std convention for the mean/std ablation.
        scale = math.sqrt(variance) + 1e-6 if config.std_normalization else 1.0
        correction = n / (n - 1) if config.baseline == "loo" else 1.0
        for i in indices:
            advantages[i] = (rewards[i] - mean) * correction / scale
    return advantages, {
        "advantage_zero_group_fraction": zero_groups / len(buckets),
        "advantage_mean": math.fsum(advantages) / len(advantages),
        "advantage_second_moment": math.fsum(a * a for a in advantages) / len(advantages),
    }
