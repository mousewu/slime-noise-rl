import json

import pytest

from noise_rl.preflight import validate_local_checkpoints, validate_local_hf_checkpoint


def make_hf_checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": 36,
                "hidden_size": 2560,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "rope_theta": 5000000,
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model-00001-of-00002.safetensors").write_bytes(b"fixture")


def test_local_checkpoint_validation(tmp_path):
    hf = tmp_path / "hf"
    megatron = tmp_path / "megatron"
    make_hf_checkpoint(hf)
    megatron.mkdir()
    (megatron / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")
    assert validate_local_checkpoints(hf, megatron) == (hf.resolve(), megatron.resolve())


def test_local_hf_checkpoint_requires_weights(tmp_path):
    hf = tmp_path / "hf"
    make_hf_checkpoint(hf)
    (hf / "model-00001-of-00002.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="weight"):
        validate_local_hf_checkpoint(hf)
