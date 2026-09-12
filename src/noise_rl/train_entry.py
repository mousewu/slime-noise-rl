"""Initialize an isolated Ray runtime with explicit project imports, then enter Slime."""

import json
import logging
import os
import runpy
import sys
import uuid
from pathlib import Path


_UNIX_SOCKET_PATH_MAX = 107
_RAY_SOCKET_SUFFIX = "session_2026-09-07_23-59-59_999999_999999/sockets/plasma_store"


def ray_temp_directory() -> str | None:
    """Return a short lexical Ray root without resolving a useful symlink.

    Ray puts Unix-domain sockets under its session directory.  Unlike ordinary
    temporary files, those have a 107-byte path limit on Linux.  A project data
    directory can be a perfectly valid TMPDIR for FastDownward yet be too long
    for Ray.  ``NOISE_RL_RAY_TMPDIR`` may therefore point to a short symlink
    (for example ``/tmp/nrl``) targeting that same large filesystem.
    """
    configured = os.environ.get("NOISE_RL_RAY_TMPDIR") or os.environ.get("TMPDIR")
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise ValueError("TMPDIR and NOISE_RL_RAY_TMPDIR must be absolute paths")
    ray_path = path / "ray"
    socket_path = ray_path / _RAY_SOCKET_SUFFIX
    if len(os.fsencode(str(socket_path))) > _UNIX_SOCKET_PATH_MAX:
        raise ValueError(
            "Ray temporary directory is too long for its Unix socket path. "
            "Keep TMPDIR for FastDownward, then set NOISE_RL_RAY_TMPDIR to a short "
            "absolute directory or symlink such as /tmp/nrl."
        )
    # Deliberately do not call resolve(): a short symlink is the intended way
    # to use a deep project filesystem while retaining a short socket pathname.
    path.mkdir(parents=True, exist_ok=True)
    return str(ray_path)


def main(entrypoint: str = "train.py"):
    import ray

    from .awm_log_mirror import start_awm_server_log_mirror
    from .ray_log_mirror import start_ray_log_mirror
    from .rollout_diagnostics import install_slime_timer_patch
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
    fully_async = entrypoint == "train_async.py"
    if fully_async:
        # Propagate this explicit mode marker into all Ray workers.  It keeps
        # the runtime-only Slime wrapper out of the normal synchronous path.
        os.environ["NOISE_RL_FULLY_ASYNC_QUEUE_METRICS"] = "1"
        from .async_queue_metrics import install_fully_async_queue_metrics

        install_fully_async_queue_metrics()
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
            "NOISE_RL_FULLY_ASYNC_QUEUE_METRICS",
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "NVTE_FUSED_ATTN",
            "NVTE_FLASH_ATTN",
            "NCCL_ALGO",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
            "CUBLAS_WORKSPACE_CONFIG",
            "SWANLAB_API_KEY",
            "SWANLAB_API_HOST",
            # Project-local bounded telemetry controls.  These must reach the
            # Ray workers that produce rollout and Slime log records.
            "NOISE_RL_SWANLAB_LOG_LEVEL",
            "NOISE_RL_SWANLAB_MAX_PENDING_LOGS",
            "NOISE_RL_SWANLAB_MAX_PENDING_METRICS",
            SWANLAB_CONFIG_ENV,
            SWANLAB_ACTOR_ENV,
        )
        if k in os.environ
    }
    runtime_env = {"env_vars": propagated, "worker_process_setup_hook": setup_worker}
    ray_address = os.environ.get("RAY_ADDRESS", "local")
    ray_options = {
        "address": ray_address,
        "runtime_env": runtime_env,
        "log_to_driver": True,
    }
    # Ray itself otherwise writes its session and object-spill files under
    # /tmp.  A user-provided TMPDIR makes one durable parent location for both
    # Ray and ALFWorld without forcing a storage choice on every deployment.
    ray_tmpdir = ray_temp_directory()
    if ray_tmpdir and ray_address == "local":
        ray_options["_temp_dir"] = ray_tmpdir
    ray.init(**ray_options)
    install_slime_timer_patch()
    if ray_address == "local":
        ray_log_mirror = start_ray_log_mirror(
            ray_tmpdir,
            audit_path=os.environ.get("NOISE_RL_RAY_DIAGNOSTICS_PATH"),
        )
    else:
        logging.getLogger(__name__).warning(
            "Ray diagnostic mirror is unavailable for remote RAY_ADDRESS=%s; "
            "inspect the Ray head node logs",
            ray_address,
        )
        ray_log_mirror = None
    swanlab_logger = None
    awm_log_mirror = None
    try:
        if swanlab_settings:
            logger_type = ray.remote(num_cpus=0)(SwanLabLogger)
            swanlab_logger = logger_type.options(name=os.environ[SWANLAB_ACTOR_ENV]).remote(swanlab_settings)
            ray.get(swanlab_logger.ready.remote())
            install_slime_logging_patch()
            install_log_mirror()
        awm_log_mirror = start_awm_server_log_mirror(
            audit_path=os.environ.get("NOISE_RL_AWM_DIAGNOSTICS_PATH")
        )
        runpy.run_path(str(script), run_name="__main__")
    finally:
        training_failed = sys.exc_info()[0] is not None
        try:
            if ray_log_mirror is not None:
                ray_log_mirror.poll_once()
            if awm_log_mirror is not None:
                # Read errors flushed by Uvicorn immediately before the driver
                # exits, while the SwanLab owner still accepts forwarded logs.
                awm_log_mirror.poll_once()
                awm_log_mirror.stop()
            # Explicit finish means "Completed" in SwanLab. On an exception,
            # leave the run unfinished so the service can classify it as an
            # interrupted/crashed experiment instead of a successful one.
            if swanlab_logger is not None and not training_failed:
                ray.get(swanlab_logger.finish.remote(), timeout=60)
        finally:
            try:
                ray.shutdown()
            finally:
                if ray_log_mirror is not None:
                    ray_log_mirror.poll_once()
                    ray_log_mirror.stop()


if __name__ == "__main__":
    main()
