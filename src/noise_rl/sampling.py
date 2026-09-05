import hashlib
import json
from dataclasses import asdict, dataclass

from .config import ExperimentConfig


def stable_seed(*parts) -> int:
    """No process-global RNG or randomized Python hash; safe across worker orderings."""
    data = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


@dataclass(frozen=True)
class SamplingPlan:
    task_id: str
    group_id: int
    rank: int
    scenario_id: int
    environment_seed: int
    policy_seed: int
    evaluation: bool = False

    def to_dict(self):
        return asdict(self)


def plan_sample(
    config: ExperimentConfig, task_id: str, group_id: int, rank: int, evaluation: bool = False
) -> SamplingPlan:
    if group_id < 0 or rank < 0 or (not evaluation and rank >= config.group_size):
        raise ValueError("Invalid sample identity")
    # Evaluation always uses independent, held-out environment seeds, regardless of training method.
    scenario = (
        rank
        if evaluation or config.method == "independent"
        else rank // (config.group_size // config.scenarios)
    )
    domain = "evaluation" if evaluation else "training"
    root_seed = config.eval_seed if evaluation else config.seed
    return SamplingPlan(
        task_id,
        group_id,
        rank,
        scenario,
        stable_seed(root_seed, domain, "environment", task_id, group_id, scenario),
        stable_seed(root_seed, domain, "policy", task_id, group_id, rank) % (2**31 - 1),
        evaluation,
    )
