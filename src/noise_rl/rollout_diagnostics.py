"""Process-local state used to explain aborted asynchronous rollouts."""

from __future__ import annotations

import importlib
import logging
from threading import Lock
from time import monotonic


_HANDLER_NAME = "noise_rl_weight_update_observer"


class WeightUpdateTracker:
    """Track Slime's update-weights timer from its normal log messages."""

    def __init__(self):
        self._lock = Lock()
        self._started_at: float | None = None
        self._last_ended_at: float | None = None
        self._last_duration_seconds: float | None = None
        self._observed = False

    def observe(self, message: str, *, now: float | None = None) -> None:
        now = monotonic() if now is None else now
        if message.startswith("Timer update_weights start"):
            with self._lock:
                self._started_at = now
                self._observed = True
        elif message.startswith("Timer update_weights end"):
            with self._lock:
                if self._started_at is not None:
                    self._last_duration_seconds = max(0.0, now - self._started_at)
                self._started_at = None
                self._last_ended_at = now
                self._observed = True

    def snapshot(self, *, now: float | None = None) -> dict[str, bool | float | None | str]:
        now = monotonic() if now is None else now
        with self._lock:
            return {
                # Timers are emitted in the current Ray process. A false value
                # does not claim that no other process is updating weights.
                "scope": "current_process_timer_log",
                "observed": self._observed,
                "active": self._started_at is not None,
                "active_seconds": (
                    max(0.0, now - self._started_at) if self._started_at is not None else None
                ),
                "last_ended_seconds_ago": (
                    max(0.0, now - self._last_ended_at)
                    if self._last_ended_at is not None
                    else None
                ),
                "last_duration_seconds": self._last_duration_seconds,
            }


_WEIGHT_UPDATE_TRACKER = WeightUpdateTracker()


def weight_update_snapshot() -> dict[str, bool | float | None | str]:
    """Return the update phase visible to the current rollout process."""
    return _WEIGHT_UPDATE_TRACKER.snapshot()


class _WeightUpdateLogObserver(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.name != "slime.observability.timer":
            return
        try:
            _WEIGHT_UPDATE_TRACKER.observe(record.getMessage())
        except Exception:
            # Diagnostics must never interfere with Slime's logger.
            return


def install_weight_update_observer() -> None:
    """Attach a handler that survives Slime's logger reconfiguration."""
    root = logging.getLogger()
    if any(getattr(handler, "_noise_rl_name", None) == _HANDLER_NAME for handler in root.handlers):
        return
    handler = _WeightUpdateLogObserver(level=logging.INFO)
    handler._noise_rl_name = _HANDLER_NAME
    root.addHandler(handler)


def install_slime_timer_patch() -> None:
    """Reinstall the observer after Slime calls ``basicConfig(force=True)``."""
    try:
        logging_utils = importlib.import_module("slime.observability.logging_utils")
    except ImportError:
        return
    original_configure = getattr(logging_utils, "configure_logger", None)
    if callable(original_configure) and not getattr(
        original_configure, "_noise_rl_weight_update_observer", False
    ):

        def configure_logger(*args, **kwargs):
            result = original_configure(*args, **kwargs)
            install_weight_update_observer()
            return result

        configure_logger._noise_rl_weight_update_observer = True
        logging_utils.configure_logger = configure_logger
    install_weight_update_observer()
