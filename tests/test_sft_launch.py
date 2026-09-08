import sys
from types import SimpleNamespace

import pytest

from noise_rl import sft_launch


def options(tmp_path):
    return SimpleNamespace(
        slime_dir="/slime",
        output=str(tmp_path / "sft"),
        gpus=8,
        tensor_parallel=2,
        batch_size=32,
        hf_checkpoint="/models/hf",
        megatron_checkpoint="/models/megatron",
        data="/data/alfworld-sft.jsonl",
        seed=42,
        save_interval=100,
        epochs=3,
        lr=5e-6,
        min_lr=5e-7,
        warmup_fraction=0.1,
        max_tokens_per_gpu=8192,
    )


def test_sft_command_uses_official_rule_based_sft_rollout_without_sglang(tmp_path, monkeypatch):
    args = options(tmp_path)
    monkeypatch.setattr(sft_launch, "model_arguments", lambda _slime: ["--rotary-base", "5000000"])

    command = sft_launch.build_command(args)

    def value(key):
        return command[command.index(key) + 1]

    assert command[:3] == [sys.executable, "-m", "noise_rl.sft_train_entry"]
    assert value("--rollout-function-path") == "slime.rollout.sft_rollout.generate_rollout"
    assert value("--input-key") == "messages"
    assert value("--loss-type") == "sft_loss"
    assert value("--num-epoch") == "3"
    assert value("--global-batch-size") == "32"
    assert value("--seed") == "42"
    assert "--debug-train-only" in command
    assert "--sequence-parallel" in command
    assert not any("sglang" in part.lower() for part in command)
    assert "--colocate" not in command


def test_sft_rejects_global_batch_not_divisible_by_data_parallelism(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.batch_size = 30
    monkeypatch.setattr(sft_launch, "model_arguments", lambda _slime: [])

    with pytest.raises(ValueError, match="Global batch"):
        sft_launch.build_command(args)


def test_sft_capability_requires_slime_native_rollout_module(tmp_path):
    slime = tmp_path / "slime"
    (slime / "slime" / "rollout").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="native SFT"):
        sft_launch.verify_sft_capability(slime)
    (slime / "slime" / "rollout" / "sft_rollout.py").write_text(
        "def generate_rollout(args, rollout_id, data_buffer, evaluation=False):\n    return []\n",
        encoding="utf-8",
    )
    sft_launch.verify_sft_capability(slime)
