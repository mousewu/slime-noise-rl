"""Initialize an isolated Ray runtime with explicit project imports, then enter Slime."""

import json
import os
import runpy
import sys
import uuid
from pathlib import Path


def main(entrypoint: str = "train.py"):
    import ray

    from .swanlab_bridge import (
        SWANLAB_ACTOR_ENV,
        SWANLAB_CONFIG_ENV,
        SwanLabLogger,
        install_log_mirror,
        install_slime_logging_patch,
        setup_worker,
    )

    slime = Path(sys.argv[1]).resolve()
    script = slime / entrypoint
    if not script.is_file():
        raise FileNotFoundError(f"Slime {entrypoint} was not found in {slime}")
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(slime))
    swanlab_settings = (
        json.loads(os.environ[SWANLAB_CONFIG_ENV]) if os.environ.get(SWANLAB_CONFIG_ENV) else None
    )
    if swanlab_settings:
        os.environ[SWANLAB_ACTOR_ENV] = f"noise_rl_swanlab_{uuid.uuid4().hex}"
    propagated = {
        k: os.environ[k]
        for k in (
            "PYTHONPATH",
            # FastDownward (called by TextWorld) and the process-isolated
            # environment runners inherit this location from the Ray worker.
            "TMPDIR",
            "NOISE_RL_RAY_TMPDIR",
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "NVTE_FUSED_ATTN",
            "NVTE_FLASH_ATTN",
            "NCCL_ALGO",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
            "CUBLAS_WORKSPACE_CONFIG",
            "SWANLAB_API_KEY",
            "SWANLAB_API_HOST",
            SWANLAB_CONFIG_ENV,
            SWANLAB_ACTOR_ENV,
        )
        if k in os.environ
    }
    runtime_env = {"env_vars": propagated}
    if swanlab_settings:
        runtime_env["worker_process_setup_hook"] = setup_worker
    ray_address = os.environ.get("RAY_ADDRESS", "local")
    ray_options = {
        "address": ray_address,
        "runtime_env": runtime_env,
        "log_to_driver": True,
    }
    # Ray itself otherwise writes its session and object-spill files under
    # /tmp.  A user-provided TMPDIR makes one durable parent location for both
    # Ray and ALFWorld without forcing a storage choice on every deployment.
    ray_tmpdir = os.environ.get("NOISE_RL_RAY_TMPDIR") or os.environ.get("TMPDIR")
    if ray_tmpdir and ray_address == "local":
        path = Path(ray_tmpdir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        ray_options["_temp_dir"] = str(path / "ray")
    ray.init(**ray_options)
    swanlab_logger = None
    try:
        if swanlab_settings:
            logger_type = ray.remote(num_cpus=0)(SwanLabLogger)
            swanlab_logger = logger_type.options(name=os.environ[SWANLAB_ACTOR_ENV]).remote(swanlab_settings)
            ray.get(swanlab_logger.ready.remote())
            install_slime_logging_patch()
            install_log_mirror()
        runpy.run_path(str(script), run_name="__main__")
    finally:
        training_failed = sys.exc_info()[0] is not None
        try:
            # Explicit finish means "Completed" in SwanLab. On an exception,
            # leave the run unfinished so the service can classify it as an
            # interrupted/crashed experiment instead of a successful one.
            if swanlab_logger is not None and not training_failed:
                ray.get(swanlab_logger.finish.remote(), timeout=60)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
