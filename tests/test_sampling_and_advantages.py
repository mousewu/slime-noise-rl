import itertools
import math
import random
from dataclasses import replace

import pytest

from noise_rl.advantages import group_advantages
from noise_rl.config import ExperimentConfig, NoiseConfig, load_config
from noise_rl.sampling import plan_sample, stable_seed


def test_seed_is_stable_and_namespaced():
    assert stable_seed("a", 12) == stable_seed("a", 12)
    assert stable_seed("a", 12) != stable_seed("a12")


def test_shared_environment_independent_policy():
    cfg = ExperimentConfig()
    plans = [plan_sample(cfg, "task", 3, i) for i in range(8)]
    assert len({p.environment_seed for p in plans}) == 2
    assert len({p.policy_seed for p in plans}) == 8
    assert len({p.environment_seed for p in plans[:4]}) == 1
    independent = replace(cfg, method="independent")
    other = [plan_sample(independent, "task", 3, i) for i in range(8)]
    assert len({p.environment_seed for p in other}) == 8
    assert [p.policy_seed for p in other] == [p.policy_seed for p in plans]


def test_evaluation_is_independent_fixed_and_heldout():
    cfg = ExperimentConfig()
    plans = [plan_sample(cfg, "t", 0, i, evaluation=True) for i in range(8)]
    assert len({p.environment_seed for p in plans}) == 8
    for i, p in enumerate(plans):
        assert p == plan_sample(replace(cfg, method="independent", seed=100), "t", 0, i, evaluation=True)
        assert p.environment_seed != plan_sample(cfg, "t", 0, i).environment_seed


def test_group_centering_and_order_independence():
    cfg = ExperimentConfig()
    plans = [plan_sample(cfg, "t", 0, i) for i in range(8)]
    rewards = [0, 0, 1, 1, 1, 1, 1, 1]
    values, stats = group_advantages(rewards, plans, cfg)
    assert values == pytest.approx([-2 / 3, -2 / 3, 2 / 3, 2 / 3, 0, 0, 0, 0])
    assert stats["advantage_zero_group_fraction"] == 0.5
    order = list(range(8))
    random.Random(10).shuffle(order)
    shuffled, _ = group_advantages([rewards[i] for i in order], [plans[i] for i in order], cfg)
    assert shuffled == pytest.approx([values[i] for i in order])


def test_distinct_tasks_never_share_baseline():
    cfg = ExperimentConfig(method="independent")
    plans = [plan_sample(cfg, t, g, i) for g, t in enumerate(["a", "b"]) for i in range(8)]
    values, _ = group_advantages([0] * 8 + [1] * 8, plans, cfg)
    assert values == [0] * 16


def test_std_ablation_matches_sample_std():
    cfg = ExperimentConfig(method="independent", baseline="mean", std_normalization=True)
    plans = [plan_sample(cfg, "t", 0, i) for i in range(8)]
    values, _ = group_advantages([0] * 4 + [1] * 4, plans, cfg)
    expected = 0.5 / (math.sqrt(2 / 7) + 1e-6)
    assert values == pytest.approx([-expected] * 4 + [expected] * 4)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_seed", "evaluation", "nan"])
def test_malformed_groups_fail_closed(damage):
    cfg = ExperimentConfig()
    plans = [plan_sample(cfg, "t", 0, i) for i in range(8)]
    rewards = [0.0] * 8
    if damage == "missing":
        plans.pop()
        rewards.pop()
    elif damage == "duplicate":
        plans[0] = plans[1]
    elif damage == "wrong_seed":
        plans[0] = replace(plans[0], environment_seed=123)
    elif damage == "evaluation":
        plans[0] = replace(plans[0], evaluation=True)
    else:
        rewards[0] = float("nan")
    with pytest.raises(ValueError):
        group_advantages(rewards, plans, cfg)


def test_conditional_loo_policy_gradient_exact_enumeration():
    # Bernoulli action a~p, shared exogenous e~q, reward=a*e.
    # Enumerate all outcomes; E[A(a-p)] must be dJ/dlogit(p)=q*p*(1-p).
    cfg = ExperimentConfig(group_size=4, scenarios=2, baseline="loo")
    plans = [plan_sample(cfg, "t", 0, i) for i in range(4)]
    p, q, expectation = 0.3, 0.7, 0.0
    for env in itertools.product([0, 1], repeat=2):
        for actions in itertools.product([0, 1], repeat=4):
            probability = math.prod(q if e else 1 - q for e in env) * math.prod(
                p if a else 1 - p for a in actions
            )
            rewards = [a * env[i // 2] for i, a in enumerate(actions)]
            adv, _ = group_advantages(rewards, plans, cfg)
            expectation += probability * sum(v * (a - p) for v, a in zip(adv, actions)) / 4
    assert expectation == pytest.approx(q * p * (1 - p))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True])
def test_invalid_probability(value):
    with pytest.raises(ValueError):
        NoiseConfig(action_drop=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": 7},
        {"scenarios": 8},
        {"max_turns": 0},
        {"baseline": "typo"},
        {"method": "typo"},
        {"retry_limit": -1},
        {"environment_processes": -1},
        {"environment_recycle_episodes": -1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_all_configs_load():
    from pathlib import Path

    files = list((Path(__file__).parents[1] / "configs").glob("*.yaml"))
    assert len(files) >= 7
    for path in files:
        assert isinstance(load_config(path), ExperimentConfig)
