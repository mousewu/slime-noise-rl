"""Project-local SwanLab bridge for Slime's distributed metric calls.

Slime is intentionally treated as an immutable dependency.  A Ray worker setup
hook replaces its metric dispatcher at runtime, preserves the original
dispatcher, and forwards scalar metrics to one project-owned logger actor.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
from numbers import Real
from pathlib import Path
from time import time
from typing import Any

SWANLAB_CONFIG_ENV = "NOISE_RL_SWANLAB_CONFIG"
SWANLAB_ACTOR_ENV = "NOISE_RL_SWANLAB_ACTOR"

_LOGGER_ACTOR = None
_FORWARD_WARNING_EMITTED = False
_LOG_FORWARD_WARNING_EMITTED = False
_LOG_HANDLER_NAME = "noise_rl_swanlab_log_mirror"
_LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}


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
        self._audit_path = Path(init_options["logdir"]) / "metric_events.jsonl"
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_audit_path = self._audit_path.parent / "forwarded_logs.jsonl"
        self._event_index = 0
        self._log_index = 0

    def ready(self) -> dict[str, str | None]:
        return {"id": getattr(self._run, "id", None)}

    def log(self, metrics: dict[str, Any]) -> None:
        values = scalar_metrics(metrics)
        if not values:
            return
        # Slime has independent train/rollout/eval counters. SwanLab has one
        # global step, so let its SDK maintain that counter (including across
        # resume) and retain Slime's native counters in the logged values.
        self._event_index += 1
        event = {"event": self._event_index, "time": time(), "metrics": values}
        with self._audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        self._swanlab.log(values)

    def log_text(self, payload: dict[str, Any]) -> None:
        """Mirror a worker log record through this actor's SwanLab-captured stdout."""
        level = str(payload.get("level", "INFO")).upper()
        logger_name = str(payload.get("logger", "root"))
        message = _redact_log_text(str(payload.get("message", "")))
        timestamp = float(payload.get("time", time()))
        self._log_index += 1
        event = {
            "event": self._log_index,
            "time": timestamp,
            "level": level,
            "logger": logger_name,
            "message": message,
        }
        with self._log_audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        # SwanLab captures stdout from the process that owns swanlab.init().
        # Split long/multiline records so neither the SDK nor the web UI truncates
        # an entire Ray exception or DetailLogger entry.
        prefix = f"[noise-rl][{level}][{logger_name}] "
        width = max(1, 900 - len(prefix))
        lines = message.splitlines() or [""]
        for line in lines:
            for offset in range(0, max(1, len(line)), width):
                print(prefix + (line[offset : offset + width] or ""), flush=True)

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


def _forward(metrics: dict[str, Any]) -> bool:
    global _FORWARD_WARNING_EMITTED
    values = scalar_metrics(metrics)
    if not values:
        return False
    if not os.environ.get(SWANLAB_ACTOR_ENV):
        return False
    try:
        import ray

        ray.get(_logger_actor().log.remote(values), timeout=30)
        return True
    except Exception:
        if not _FORWARD_WARNING_EMITTED:
            logging.getLogger(__name__).exception(
                "SwanLab metric forwarding failed; training will continue without further warning"
            )
            _FORWARD_WARNING_EMITTED = True
        return False


def report_metrics(metrics: dict[str, Any]) -> bool:
    """Forward project-owned scalar metrics when SwanLab tracking is enabled."""
    return _forward(metrics)


def _redact_log_text(value: str) -> str:
    """Do not mirror the credentials that may appear in child-process output."""
    for name in ("SWANLAB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, f"<{name}_REDACTED>")
    return value


def _forward_log(payload: dict[str, Any]) -> None:
    """Submit a worker log line without delaying model, environment, or Ray work."""
    global _LOG_FORWARD_WARNING_EMITTED
    if not os.environ.get(SWANLAB_ACTOR_ENV):
        return
    try:
        _logger_actor().log_text.remote(payload)
    except Exception:
        # Do not use logging here: the caller is a logging handler and would
        # recursively invoke this path. The local Ray log remains authoritative.
        if not _LOG_FORWARD_WARNING_EMITTED:
            _LOG_FORWARD_WARNING_EMITTED = True


class _SwanLabLogMirror(logging.Handler):
    """Forward standard Python logs from a Ray process to the SwanLab owner."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(("noise_rl.swanlab_bridge", "swanlab")):
            return
        try:
            message = self.format(record)
        except Exception:
            return
        _forward_log(
            {
                "time": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            }
        )


def install_log_mirror() -> None:
    """Attach one INFO+ handler after Slime has configured this process's root logger."""
    root = logging.getLogger()
    if any(getattr(handler, "_noise_rl_name", None) == _LOG_HANDLER_NAME for handler in root.handlers):
        return
    configured_level = os.environ.get("NOISE_RL_SWANLAB_LOG_LEVEL", "INFO").upper()
    handler = _SwanLabLogMirror(level=_LOG_LEVELS.get(configured_level, logging.INFO))
    handler._noise_rl_name = _LOG_HANDLER_NAME
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)


def install_slime_logging_patch() -> None:
    """Preserve Slime's logger and add SwanLab forwarding in this process."""
    logging_utils = importlib.import_module("slime.observability.logging_utils")
    original_log = logging_utils.log
    if not getattr(original_log, "_noise_rl_swanlab_bridge", False):

        def log(args, metrics, step_key: str):
            original_log(args, metrics, step_key)
            _forward(metrics)

        log._noise_rl_swanlab_bridge = True
        logging_utils.log = log

    original_configure = getattr(logging_utils, "configure_logger", None)
    if callable(original_configure) and not getattr(original_configure, "_noise_rl_swanlab_log_mirror", False):

        def configure_logger(*args, **kwargs):
            result = original_configure(*args, **kwargs)
            # Slime uses logging.basicConfig(force=True), which removes handlers
            # installed before it configures a worker's root logger.
            install_log_mirror()
            return result

        configure_logger._noise_rl_swanlab_log_mirror = True
        logging_utils.configure_logger = configure_logger
    install_log_mirror()


def setup_worker() -> None:
    """Ray runtime-env hook executed before Slime actors and tasks are loaded."""
    if os.environ.get(SWANLAB_CONFIG_ENV):
        install_slime_logging_patch()
