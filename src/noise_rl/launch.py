"""Safe single-node launcher: no process killing, automatic downloads, or checkpoint overwrites."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path

from . import SLIME_COMMIT
from .config import load_config
from .data import atomic_json, read_records
from .swanlab_bridge import SWANLAB_CONFIG_ENV

_TRACKING_OPTIONS = {
    "use_swanlab",
    "swanlab_project",
    "swanlab_workspace",
    "swanlab_experiment_name",
    "swanlab_description",
    "swanlab_group",
    "swanlab_tags",
    "swanlab_mode",
    "swanlab_logdir",
}


def swanlab_metadata(args, output: Path, run_id: str) -> dict:
    """Build non-secret tracking metadata persisted with the training run."""
    return {
        "run_id": run_id,
        "project": args.swanlab_project,
        "workspace": args.swanlab_workspace,
        "experiment_name": args.swanlab_experiment_name or output.name,
        "description": args.swanlab_description,
        "group": args.swanlab_group,
        "tags": list(args.swanlab_tags),
        "mode": args.swanlab_mode,
        "logdir": str(
            Path(args.swanlab_logdir).expanduser().resolve() if args.swanlab_logdir else output / "swanlab"
        ),
    }


def swanlab_runtime_config(tracking: dict, experiment: dict) -> dict:
    """Translate persisted metadata to documented swanlab.init options."""
    options = {key: value for key, value in tracking.items() if key != "run_id"}
    options.update(id=tracking["run_id"], resume="allow", config=experiment)
    return options


def verify_slime(slime_dir):
    actual = subprocess.check_output(["git", "-C", str(slime_dir), "rev-parse", "HEAD"], text=True).strip()
    if actual != SLIME_COMMIT:
        raise ValueError(f"Expected Slime {SLIME_COMMIT}, found {actual}; port and re-test before updating")
    # Untracked files are allowed; modified tracked source is not a verified dependency.
    dirty = subprocess.check_output(
        ["git", "-C", str(slime_dir), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise ValueError("Slime tracked source is modified; use a clean pinned checkout")
    return actual


def model_arguments(slime_dir):
    script = Path(slime_dir) / "scripts/models/qwen3-4B-Instruct-2507.sh"
    # Positional arguments, never interpolate paths into shell source code.
    output = subprocess.check_output(
        ["bash", "-c", 'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"', "model-config", str(script)]
    )
    return [x.decode() for x in output.split(b"\0") if x]


def build_command(args, config):
    slime = Path(args.slime_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    batch = args.batch_size * config.group_size
    if args.gpus < 1 or args.tensor_parallel < 1 or args.gpus % args.tensor_parallel:
        raise ValueError("GPU count must be divisible by tensor parallelism")
    if args.gpus % args.engine_gpus:
        raise ValueError("GPU count must be divisible by GPUs per rollout engine")
    if batch % (args.gpus // args.tensor_parallel):
        raise ValueError("Global batch must be divisible by data-parallel size")
    command = [sys.executable, "-m", "noise_rl.train_entry", str(slime)] + model_arguments(slime)
    command += [
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(args.gpus),
        "--colocate",
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
        "--save-hf",
        str(output / "hf" / "rollout_{rollout_id}"),
        "--prompt-data",
        str(Path(args.data).expanduser().resolve()),
        "--input-key",
        "prompt",
        "--metadata-key",
        "metadata",
        "--rollout-shuffle",
        "--data-source-path",
        "noise_rl.data.NoiseDataSource",
        "--custom-generate-function-path",
        "noise_rl.slime_hooks.generate",
        "--custom-reward-post-process-path",
        "noise_rl.slime_hooks.reward_postprocess",
        "--eval-function-path",
        "noise_rl.slime_hooks.evaluate_rollout",
        "--custom-config-path",
        str(output / "runtime_config.json"),
        "--num-rollout",
        str(args.num_rollout),
        "--rollout-batch-size",
        str(args.batch_size),
        "--over-sampling-batch-size",
        str(args.batch_size),
        "--n-samples-per-prompt",
        str(config.group_size),
        "--global-batch-size",
        str(batch),
        "--num-steps-per-rollout",
        "1",
        "--rollout-max-response-len",
        str(config.max_generated_tokens),
        "--rollout-max-context-len",
        str(config.max_context_tokens),
        "--rollout-temperature",
        "0.8",
        "--rollout-top-p",
        "1",
        "--rollout-top-k",
        "-1",
        "--rollout-seed",
        str(config.seed),
        "--seed",
        str(config.seed),
        "--advantage-estimator",
        "grpo",
        "--disable-grpo-std-normalization",
        "--kl-coef",
        "0",
        "--entropy-coef",
        "0",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.2",
        "--optimizer",
        "adam",
        "--lr",
        str(args.lr),
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
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
        "--balance-data",
        "--rollout-num-gpus-per-engine",
        str(args.engine_gpus),
        "--sglang-server-concurrency",
        str(max(1, config.concurrency // (args.gpus // args.engine_gpus))),
        "--sglang-mem-fraction-static",
        "0.55",
        "--sglang-attention-backend",
        "flashinfer",
        "--sglang-context-length",
        str(config.max_context_tokens),
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
    if getattr(args, "deterministic", False):
        command += ["--sglang-enable-deterministic-inference", "--deterministic-mode"]
    if args.debug_rollout_only:
        command.append("--debug-rollout-only")
        command += ["--save-debug-rollout-data", str(output / "debug" / "rollout_{rollout_id}.pt")]
    if args.eval_data:
        command += [
            "--eval-interval",
            str(args.eval_interval),
            "--eval-prompt-data",
            "heldout",
            str(Path(args.eval_data).expanduser().resolve()),
            "--n-samples-per-eval-prompt",
            str(args.eval_repeats),
            "--eval-max-response-len",
            str(config.max_generated_tokens),
            "--eval-temperature",
            "0.8",
            "--eval-top-p",
            "1",
            "--eval-top-k",
            "-1",
        ]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slime-dir", default=os.environ.get("SLIME_DIR"), required=not os.environ.get("SLIME_DIR")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval-data")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--megatron-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--engine-gpus", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-rollout", type=int, default=300)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-repeats", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=9216)
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
        "--deterministic",
        action="store_true",
        help="Enable the Slime deterministic recipe; verify GPU kernel support first",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate source/data and print command without starting Ray"
    )
    parser.add_argument("--debug-rollout-only", action="store_true")
    args = parser.parse_args(argv)
    for key in (
        "engine_gpus",
        "batch_size",
        "num_rollout",
        "save_interval",
        "eval_interval",
        "eval_repeats",
        "max_tokens_per_gpu",
    ):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.lr <= 0:
        parser.error("lr must be positive")
    config = replace(
        load_config(args.config),
        seed=args.seed,
        trace_dir=str(Path(args.output).expanduser().resolve() / "traces"),
    )
    if args.max_tokens_per_gpu < config.max_context_tokens:
        raise ValueError("max-tokens-per-gpu must accommodate one full trajectory context")
    verify_slime(args.slime_dir)
    training = read_records(args.data)
    if any(r["metadata"]["task"].get("split") != "train" for r in training):
        raise ValueError("Training tasks must be from split=train")
    if args.eval_data:
        evaluation = read_records(args.eval_data)
        ids = {r["metadata"]["task"]["id"] for r in training}
        if any(
            r["metadata"]["task"].get("split") == "train" or r["metadata"]["task"]["id"] in ids
            for r in evaluation
        ):
            raise ValueError("Train/eval overlap")
    command = build_command(args, config)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    from .preflight import check_gpu_runtime

    check_gpu_runtime(args.hf_checkpoint, args.megatron_checkpoint, args.gpus)
    output = Path(args.output).expanduser().resolve()
    invariants = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume", "dry_run", "num_rollout", *_TRACKING_OPTIONS}
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
        "slime_commit": SLIME_COMMIT,
        "config": config.to_dict(),
        "command": command,
        "training_options": invariants,
    }
    if tracking:
        metadata["swanlab"] = tracking
    elif previous_tracking:
        metadata["swanlab"] = previous_tracking
    if args.resume:
        if (
            old["config"] != metadata["config"]
            or old["slime_commit"] != SLIME_COMMIT
            or old.get("training_options") != invariants
        ):
            raise ValueError("Cannot resume with a different configuration or Slime version")
        if not (output / "checkpoints/latest_checkpointed_iteration.txt").is_file():
            raise ValueError("--resume requires an existing training checkpoint")
        if tracking and not previous_tracking:
            atomic_json(output / "run.json", metadata)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output directory is not empty; choose a new run name or pass --resume")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "run.json", metadata)
        atomic_json(output / "runtime_config.json", {"noise_rl": config.to_dict()})
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(Path(__file__).resolve().parents[1]),
                str(Path(args.slime_dir).resolve()),
                env.get("PYTHONPATH"),
            ],
        )
    )
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("NVTE_FUSED_ATTN", "0")
    env.setdefault("NVTE_FLASH_ATTN", "1")
    if args.deterministic:
        env.update(NCCL_ALGO="Ring", NVTE_ALLOW_NONDETERMINISTIC_ALGO="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if tracking:
        experiment = {
            "slime_commit": SLIME_COMMIT,
            "noise_rl": config.to_dict(),
            "training": invariants,
        }
        env[SWANLAB_CONFIG_ENV] = json.dumps(swanlab_runtime_config(tracking, experiment), sort_keys=True)
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
