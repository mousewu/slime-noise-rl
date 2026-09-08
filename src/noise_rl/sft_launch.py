"""Safe local-only launcher for ALFWorld expert warm-start SFT on Slime."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

from .data import atomic_json
from .launch import (
    _TRACKING_OPTIONS,
    configure_megatron_lm_path,
    model_arguments,
    swanlab_metadata,
    swanlab_runtime_config,
    verify_slime,
)
from .preflight import check_sft_gpu_runtime, validate_local_checkpoints
from .sft_data import validate_sft_records
from .swanlab_bridge import SWANLAB_CONFIG_ENV


def verify_sft_capability(slime_dir) -> None:
    """Fail before Ray starts when a Slime checkout lacks its native SFT rollout."""
    module = Path(slime_dir).expanduser().resolve() / "slime" / "rollout" / "sft_rollout.py"
    if not module.is_file() or "def generate_rollout" not in module.read_text(encoding="utf-8"):
        raise FileNotFoundError(
            "This Slime checkout has no slime.rollout.sft_rollout.generate_rollout; "
            "use a checkout with Slime's native SFT support."
        )


def build_command(args) -> list[str]:
    """Build Slime's official rule-based SFT recipe without an inference server."""
    slime = Path(args.slime_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if args.gpus < 1 or args.tensor_parallel < 1 or args.gpus % args.tensor_parallel:
        raise ValueError("GPU count must be divisible by tensor parallelism")
    data_parallel = args.gpus // args.tensor_parallel
    if args.batch_size % data_parallel:
        raise ValueError("Global batch must be divisible by SFT data parallelism")
    command = [sys.executable, "-m", "noise_rl.sft_train_entry", str(slime)] + model_arguments(slime)
    command += [
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(args.gpus),
        "--seed",
        str(args.seed),
        "--hf-checkpoint",
        str(Path(args.hf_checkpoint).expanduser().resolve()),
        "--ref-load",
        str(Path(args.megatron_checkpoint).expanduser().resolve()),
        "--load",
        str(output / "checkpoints"),
        "--save",
        str(output / "checkpoints"),
        "--save-interval",
        str(args.save_interval),
        "--rollout-function-path",
        "slime.rollout.sft_rollout.generate_rollout",
        "--prompt-data",
        str(Path(args.data).expanduser().resolve()),
        "--input-key",
        "messages",
        "--metadata-key",
        "metadata",
        "--rollout-shuffle",
        "--num-epoch",
        str(args.epochs),
        "--rollout-batch-size",
        str(args.batch_size),
        "--global-batch-size",
        str(args.batch_size),
        "--loss-type",
        "sft_loss",
        "--calculate-per-token-loss",
        "--disable-compute-advantages-and-returns",
        "--debug-train-only",
        "--optimizer",
        "adam",
        "--lr",
        str(args.lr),
        "--lr-decay-style",
        "cosine",
        "--min-lr",
        str(args.min_lr),
        "--lr-warmup-fraction",
        str(args.warmup_fraction),
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--tensor-model-parallel-size",
        str(args.tensor_parallel),
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        str(args.max_tokens_per_gpu),
        "--bf16",
        "--attention-dropout",
        "0",
        "--hidden-dropout",
        "0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        "flash",
    ]
    if args.tensor_parallel > 1:
        command.append("--sequence-parallel")
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slime-dir", default=os.environ.get("SLIME_DIR"), required=not os.environ.get("SLIME_DIR")
    )
    parser.add_argument("--data", required=True, help="Verified JSONL from noise_rl.sft_data")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--megatron-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--min-lr", type=float, default=5e-7)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=8192)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="slime-noise-rl")
    parser.add_argument("--swanlab-workspace")
    parser.add_argument("--swanlab-experiment-name")
    parser.add_argument("--swanlab-description")
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tags", nargs="*", default=[])
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "local"), default="online")
    parser.add_argument("--swanlab-logdir")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate local data and print the Slime command without starting Ray"
    )
    return parser


def _validate_arguments(args, parser: argparse.ArgumentParser) -> None:
    for name in ("gpus", "tensor_parallel", "batch_size", "epochs", "save_interval", "max_tokens_per_gpu"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.lr <= 0 or args.min_lr <= 0:
        parser.error("--lr and --min-lr must be positive")
    if args.min_lr > args.lr:
        parser.error("--min-lr must not exceed --lr")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("--warmup-fraction must be in [0, 1)")
    if args.max_tokens_per_gpu < 1024:
        parser.error("--max-tokens-per-gpu must be at least 1024")


def main(argv=None) -> None:
    megatron_lm_dir = configure_megatron_lm_path()
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_arguments(args, parser)
    slime_commit = verify_slime(args.slime_dir, "train_async.py")
    verify_sft_capability(args.slime_dir)
    validate_local_checkpoints(args.hf_checkpoint, args.megatron_checkpoint)
    dataset = validate_sft_records(args.data)
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return

    check_sft_gpu_runtime(args.hf_checkpoint, args.megatron_checkpoint, args.gpus)
    output = Path(args.output).expanduser().resolve()
    invariants = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume", "dry_run", *_TRACKING_OPTIONS}
    }
    old = None
    previous_tracking = None
    if args.resume:
        old = json.loads((output / "run.json").read_text(encoding="utf-8"))
        previous_tracking = old.get("swanlab")
    tracking = None
    if args.use_swanlab:
        run_id = previous_tracking["run_id"] if previous_tracking else uuid.uuid4().hex
        tracking = swanlab_metadata(args, output, run_id)
        if previous_tracking and tracking != previous_tracking:
            raise ValueError("Cannot resume with different SwanLab project, experiment, or storage settings")
    metadata = {
        "schema": 1,
        "phase": "alfworld_expert_sft",
        "slime_commit": slime_commit,
        "dataset": dataset,
        "training_options": invariants,
        "command": command,
    }
    if tracking:
        metadata["swanlab"] = tracking
    elif previous_tracking:
        metadata["swanlab"] = previous_tracking
    if args.resume:
        expected = {key: old.get(key) for key in ("phase", "slime_commit", "dataset", "training_options")}
        actual = {key: metadata.get(key) for key in ("phase", "slime_commit", "dataset", "training_options")}
        if expected != actual:
            raise ValueError("Cannot resume with different SFT data, Slime version, or training options")
        if not (output / "checkpoints/latest_checkpointed_iteration.txt").is_file():
            raise ValueError("--resume requires an existing SFT checkpoint")
        if tracking and not previous_tracking:
            atomic_json(output / "run.json", metadata)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output directory is not empty; choose a new run name or pass --resume")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "run.json", metadata)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").split(os.pathsep)
    if megatron_lm_dir:
        existing_pythonpath = [item for item in existing_pythonpath if item != megatron_lm_dir]
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                megatron_lm_dir,
                str(Path(__file__).resolve().parents[1]),
                str(Path(args.slime_dir).resolve()),
                *existing_pythonpath,
            ],
        )
    )
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("NVTE_FUSED_ATTN", "0")
    env.setdefault("NVTE_FLASH_ATTN", "1")
    env["NOISE_RL_RAY_DIAGNOSTICS_PATH"] = str(output / "ray_diagnostics.log")
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    if tracking:
        experiment = {
            "phase": "alfworld_expert_sft",
            "slime_commit": slime_commit,
            "dataset": dataset,
            "training": invariants,
        }
        env[SWANLAB_CONFIG_ENV] = json.dumps(swanlab_runtime_config(tracking, experiment), sort_keys=True)
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
