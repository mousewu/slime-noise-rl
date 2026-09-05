import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from noise_rl.config import ExperimentConfig
from noise_rl.launch import build_command, swanlab_metadata, swanlab_runtime_config


def options(tmp_path):
    return SimpleNamespace(
        slime_dir=os.environ.get("SLIME_SOURCE_PATH", ""),
        output=str(tmp_path / "run"),
        gpus=8,
        tensor_parallel=2,
        engine_gpus=2,
        batch_size=16,
        hf_checkpoint="/models/hf",
        megatron_checkpoint="/models/megatron",
        data="/data/train.jsonl",
        eval_data="/data/eval.jsonl",
        save_interval=25,
        num_rollout=300,
        lr=1e-6,
        max_tokens_per_gpu=9216,
        debug_rollout_only=False,
        eval_interval=25,
        eval_repeats=4,
    )


@pytest.mark.integration
def test_launch_budgets_hooks_and_checkpoint_placeholders(tmp_path):
    args = options(tmp_path)
    if not args.slime_dir:
        pytest.skip("Set SLIME_SOURCE_PATH")
    command = build_command(args, ExperimentConfig())

    def value(key):
        return command[command.index(key) + 1]

    assert int(value("--rollout-batch-size")) * int(value("--n-samples-per-prompt")) == 128
    assert value("--global-batch-size") == "128"
    assert value("--over-sampling-batch-size") == value("--rollout-batch-size")
    assert value("--custom-reward-post-process-path") == "noise_rl.slime_hooks.reward_postprocess"
    assert value("--save-hf").format(rollout_id=3).endswith("rollout_3")
    assert value("--rotary-base") == "5000000"
    assert "--apply-chat-template" not in command
    assert not Path(args.output).exists()
    args.deterministic = True
    command = build_command(args, ExperimentConfig())
    assert "--sglang-enable-deterministic-inference" in command
    assert "--deterministic-mode" in command
    args.debug_rollout_only = True
    command = build_command(args, ExperimentConfig())
    assert value("--save-debug-rollout-data").format(rollout_id="eval_3").endswith("rollout_eval_3.pt")


def test_invalid_parallelism_rejected_before_launch(tmp_path):
    args = options(tmp_path)
    args.gpus = 7
    with pytest.raises(ValueError, match="divisible"):
        build_command(args, ExperimentConfig())


def test_swanlab_config_is_project_owned_and_contains_no_key(tmp_path):
    args = SimpleNamespace(
        swanlab_project="agentic-noise-rl",
        swanlab_workspace="lab",
        swanlab_experiment_name=None,
        swanlab_description="matched scenarios",
        swanlab_group="matched-loo",
        swanlab_tags=["alfworld", "seed-42"],
        swanlab_mode="offline",
        swanlab_logdir=None,
    )
    output = tmp_path / "matched-s42"
    tracking = swanlab_metadata(args, output, "stable-run-id")
    runtime = swanlab_runtime_config(tracking, {"seed": 42})

    assert tracking["experiment_name"] == "matched-s42"
    assert tracking["logdir"] == str(output / "swanlab")
    assert runtime["id"] == "stable-run-id" and runtime["resume"] == "allow"
    assert runtime["config"] == {"seed": 42}
    assert not any("key" in name.lower() for name in runtime)
