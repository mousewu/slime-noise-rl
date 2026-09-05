import importlib
import json
from pathlib import Path


def check_gpu_runtime(hf_checkpoint, megatron_checkpoint, gpus):
    for name in ("torch", "ray", "sglang", "megatron.core", "transformer_engine", "flashinfer"):
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
    model = json.loads((Path(hf_checkpoint) / "config.json").read_text(encoding="utf-8"))
    expected = {
        "model_type": "qwen3",
        "num_hidden_layers": 36,
        "hidden_size": 2560,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rope_theta": 5000000,
    }
    if any(model.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint must match Qwen3-4B-Instruct-2507 architecture/rope settings")
    if not (Path(megatron_checkpoint) / "latest_checkpointed_iteration.txt").is_file():
        raise FileNotFoundError("Convert the initial HF model to a Megatron checkpoint first")
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
