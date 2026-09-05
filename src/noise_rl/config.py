from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class NoiseConfig:
    action_drop: float = 0.15
    observation_loss: float = 0.10
    burst_probability: float = 0.0
    burst_length: int = 3

    def __post_init__(self):
        for name in ("action_drop", "observation_loss", "burst_probability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a probability in [0, 1]")
        if type(self.burst_length) is not int or self.burst_length < 1:
            raise ValueError("burst_length must be a positive integer")


@dataclass(frozen=True)
class ExperimentConfig:
    # 'coupled_prompt' isolates coupling from conditional reward centering.
    method: str = "matched"
    seed: int = 42
    eval_seed: int = 20260904
    group_size: int = 8
    scenarios: int = 2
    baseline: str = "loo"
    std_normalization: bool = False
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    max_turns: int = 40
    max_tool_calls: int = 50
    max_context_tokens: int = 8192
    max_generated_tokens: int = 2048
    max_tokens_per_turn: int = 96
    request_timeout: float = 180.0
    concurrency: int = 16
    retry_limit: int = 0
    trace_dir: str | None = None

    def __post_init__(self):
        if type(self.seed) is not int or type(self.eval_seed) is not int:
            raise ValueError("seed and eval_seed must be integers")
        if self.method not in {"independent", "matched", "coupled_prompt"}:
            raise ValueError(f"Unknown method: {self.method}")
        if self.baseline not in {"mean", "loo"}:
            raise ValueError("baseline must be mean or loo")
        for name in (
            "group_size",
            "scenarios",
            "max_turns",
            "max_tool_calls",
            "max_context_tokens",
            "max_generated_tokens",
            "max_tokens_per_turn",
            "concurrency",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.group_size < 2 or self.group_size % self.scenarios:
            raise ValueError("group_size must be >=2 and divisible by scenarios")
        if self.method == "matched" and self.group_size // self.scenarios < 2:
            raise ValueError("Matched groups require at least two policy samples per scenario")
        if self.max_context_tokens <= self.max_tokens_per_turn:
            raise ValueError("Context must be larger than a single generation")
        if type(self.retry_limit) is not int or self.retry_limit < 0:
            raise ValueError("retry_limit must be a nonnegative integer")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if type(self.std_normalization) is not bool:
            raise ValueError("std_normalization must be boolean")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if "noise" in value:
            value["noise"] = NoiseConfig(**value["noise"])
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or set(value) != {"noise_rl"}:
        raise ValueError("Config must contain exactly the top-level key 'noise_rl'")
    return ExperimentConfig.from_dict(value["noise_rl"])


def config_from_args(args) -> ExperimentConfig:
    if not hasattr(args, "noise_rl"):
        raise ValueError("Pass --custom-config-path configs/<experiment>.yaml to Slime")
    return ExperimentConfig.from_dict(args.noise_rl)
