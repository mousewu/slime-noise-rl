import importlib
import json
from pathlib import Path

EXPECTED_MODEL_CONFIG = {
    "model_type": "qwen3",
    "num_hidden_layers": 36,
    "hidden_size": 2560,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rope_theta": 5000000,
}


def validate_local_hf_checkpoint(hf_checkpoint) -> Path:
    """Validate the fixed model without consulting a model hub."""
    path = Path(hf_checkpoint).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise NotADirectoryError(f"HF checkpoint must be a local directory: {path}")
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Local HF checkpoint is missing: {', '.join(missing)}")
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "vocab.json")
    if not any((path / name).is_file() for name in tokenizer_files):
        raise FileNotFoundError("Local HF checkpoint has no tokenizer vocabulary file")
    if not any(file.is_file() for pattern in ("*.safetensors", "*.bin") for file in path.glob(pattern)):
        raise FileNotFoundError("Local HF checkpoint has no model weight files")
    model = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if any(model.get(key) != value for key, value in EXPECTED_MODEL_CONFIG.items()):
        raise ValueError("Checkpoint must match Qwen3-4B-Instruct-2507 architecture/rope settings")
    return path


def validate_local_megatron_checkpoint(megatron_checkpoint) -> Path:
    """Require a converted checkpoint; conversion is an explicit local operation."""
    path = Path(megatron_checkpoint).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise NotADirectoryError(f"Megatron checkpoint must be a local directory: {path}")
    if not (path / "latest_checkpointed_iteration.txt").is_file():
        raise FileNotFoundError("Convert the local HF model to a Megatron checkpoint first")
    return path


def validate_local_checkpoints(hf_checkpoint, megatron_checkpoint):
    return (
        validate_local_hf_checkpoint(hf_checkpoint),
        validate_local_megatron_checkpoint(megatron_checkpoint),
    )


def check_gpu_runtime(hf_checkpoint, megatron_checkpoint, gpus):
    validate_local_checkpoints(hf_checkpoint, megatron_checkpoint)
    for name in (
        "torch",
        "ray",
        "sglang",
        "megatron.core",
        "megatron.training",
        "transformer_engine",
        "flashinfer",
    ):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise RuntimeError(
                f"Missing GPU runtime dependency: {name}; follow Slime's pinned setup"
            ) from exc
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < gpus:
        raise RuntimeError(f"Need {gpus} visible CUDA GPUs; local CPU tests do not verify training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This launch recipe requires BF16 support")
    print(
        json.dumps(
            {
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpus)],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "warning": "Preflight checks compatibility prerequisites, not end-to-end kernel correctness.",
            }
        )
    )


def check_sft_gpu_runtime(hf_checkpoint, megatron_checkpoint, gpus):
    """Validate the Slime/Megatron dependencies needed by offline SFT only.

    SFT uses Slime's rule-based ``sft_rollout`` to construct token/loss masks.
    It does not start SGLang or require FlashInfer, so keeping this check
    separate lets a fresh warm-start run fail only on dependencies it uses.
    """
    validate_local_checkpoints(hf_checkpoint, megatron_checkpoint)
    for name in ("torch", "ray", "megatron.core", "megatron.training", "transformer_engine"):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise RuntimeError(
                f"Missing SFT GPU runtime dependency: {name}; follow Slime's pinned setup"
            ) from exc
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < gpus:
        raise RuntimeError(f"Need {gpus} visible CUDA GPUs; local CPU tests do not verify training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This SFT recipe requires BF16 support")
    print(
        json.dumps(
            {
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpus)],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "mode": "offline_sft",
                "warning": "Preflight checks dependencies, not end-to-end Megatron kernel correctness.",
            }
        )
    )
