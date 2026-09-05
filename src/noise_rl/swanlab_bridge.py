"""Project-local SwanLab bridge for Slime's distributed metric calls.

Slime is intentionally treated as an immutable dependency.  A Ray worker setup
hook replaces its metric dispatcher at runtime, preserves the original
dispatcher, and forwards scalar metrics to one project-owned logger actor.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
from numbers import Real
from typing import Any

SWANLAB_CONFIG_ENV = "NOISE_RL_SWANLAB_CONFIG"
SWANLAB_ACTOR_ENV = "NOISE_RL_SWANLAB_ACTOR"

_LOGGER_ACTOR = None
_FORWARD_WARNING_EMITTED = False


def _scalar(value: Any) -> int | float | None:
    """Convert framework scalar types without transporting tensors through Ray."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Real):
        result = value
    elif hasattr(value, "item"):
        try:
            result = value.item()
        except (RuntimeError, TypeError, ValueError):
            return None
        if isinstance(result, bool):
            return int(result)
        if not isinstance(result, Real):
            return None
    else:
        return None
    result = float(result)
    if not math.isfinite(result):
        return None
    return int(result) if result.is_integer() and isinstance(value, int) else result


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, int | float]:
    """Keep only finite scalar metrics accepted by SwanLab."""
    converted = {}
    for key, value in metrics.items():
        scalar = _scalar(value)
        if scalar is not None:
            converted[str(key)] = scalar
    return converted


class SwanLabLogger:
    """Single-process SwanLab owner used as a Ray actor."""

    def __init__(self, settings: dict[str, Any]):
        try:
            import swanlab
        except ImportError as exc:  # pragma: no cover - exercised on the GPU host
            raise RuntimeError("SwanLab is enabled; install this project with the 'tracking' extra") from exc

        self._swanlab = swanlab
        init_keys = (
            "project",
            "workspace",
            "experiment_name",
            "description",
            "group",
            "tags",
            "mode",
            "logdir",
            "id",
            "resume",
        )
        init_options = {key: settings[key] for key in init_keys if settings.get(key) is not None}
        init_options["config"] = settings.get("config", {})
        self._run = swanlab.init(**init_options)

    def ready(self) -> dict[str, str | None]:
        return {"id": getattr(self._run, "id", None)}

    def log(self, metrics: dict[str, Any]) -> None:
        values = scalar_metrics(metrics)
        if not values:
            return
        # Slime has independent train/rollout/eval counters. SwanLab has one
        # global step, so let its SDK maintain that counter (including across
        # resume) and retain Slime's native counters in the logged values.
        self._swanlab.log(values)

    def finish(self) -> None:
        self._swanlab.finish()


def _logger_actor():
    global _LOGGER_ACTOR
    if _LOGGER_ACTOR is None:
        import ray

        name = os.environ.get(SWANLAB_ACTOR_ENV)
        if not name:
            raise RuntimeError(f"{SWANLAB_ACTOR_ENV} is not set")
        _LOGGER_ACTOR = ray.get_actor(name)
    return _LOGGER_ACTOR


def _forward(metrics: dict[str, Any]) -> None:
    global _FORWARD_WARNING_EMITTED
    values = scalar_metrics(metrics)
    if not values:
        return
    try:
        import ray

        ray.get(_logger_actor().log.remote(values), timeout=30)
    except Exception:
        if not _FORWARD_WARNING_EMITTED:
            logging.getLogger(__name__).exception(
                "SwanLab metric forwarding failed; training will continue without further warning"
            )
            _FORWARD_WARNING_EMITTED = True


def install_slime_logging_patch() -> None:
    """Preserve Slime's logger and add SwanLab forwarding in this process."""
    logging_utils = importlib.import_module("slime.observability.logging_utils")
    original = logging_utils.log
    if getattr(original, "_noise_rl_swanlab_bridge", False):
        return

    def log(args, metrics, step_key: str):
        original(args, metrics, step_key)
        _forward(metrics)

    log._noise_rl_swanlab_bridge = True
    logging_utils.log = log


def setup_worker() -> None:
    """Ray runtime-env hook executed before Slime actors and tasks are loaded."""
    if os.environ.get(SWANLAB_CONFIG_ENV):
        install_slime_logging_patch()
