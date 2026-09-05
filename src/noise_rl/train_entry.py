"""Initialize an isolated Ray runtime with explicit project imports, then enter Slime."""

import json
import os
import runpy
import sys
import uuid
from pathlib import Path


def main():
    import ray

    from .swanlab_bridge import (
        SWANLAB_ACTOR_ENV,
        SWANLAB_CONFIG_ENV,
        SwanLabLogger,
        install_slime_logging_patch,
        setup_worker,
    )

    slime = Path(sys.argv[1]).resolve()
    sys.argv = [str(slime / "train.py"), *sys.argv[2:]]
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
    ray.init(
        address=os.environ.get("RAY_ADDRESS", "local"),
        runtime_env=runtime_env,
        log_to_driver=True,
    )
    swanlab_logger = None
    try:
        if swanlab_settings:
            logger_type = ray.remote(num_cpus=0)(SwanLabLogger)
            swanlab_logger = logger_type.options(name=os.environ[SWANLAB_ACTOR_ENV]).remote(swanlab_settings)
            ray.get(swanlab_logger.ready.remote())
            install_slime_logging_patch()
        runpy.run_path(str(slime / "train.py"), run_name="__main__")
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
