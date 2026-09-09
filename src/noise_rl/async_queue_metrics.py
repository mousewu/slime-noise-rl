"""Runtime-only metrics for Slime's fully-async rollout queue.

Slime keeps its completed-group queue private behind its own concurrency and
top-up policy. We leave its source untouched, but replace its worker class *at runtime* before
the rollout function is resolved.  The replacement preserves the upstream
implementation and only observes task launch/completion and queue draining.
"""

from __future__ import annotations

import importlib
import logging
import threading
from statistics import mean
from time import monotonic
from typing import Any


logger = logging.getLogger(__name__)
_PATCH_MARKER = "_noise_rl_async_queue_metrics_installed"


class AsyncGroupTelemetry:
    """Thread-safe lifecycle counters for one upstream async rollout worker."""

    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight: dict[int, float] = {}
        self._ready_at: dict[int, float] = {}
        self._started_total = 0
        self._completed_total = 0
        self._delivered_total = 0
        self._aborted_total = 0
        self._requeued_total = 0
        self._failed_total = 0
        self._pending_high_water = 0
        self._last_report_at = monotonic()
        self._last_report_counts = (0, 0, 0)

    def started(self, group_id: int, *, now: float | None = None) -> None:
        now = monotonic() if now is None else now
        with self._lock:
            self._in_flight[group_id] = now
            self._started_total += 1

    def finished(self, group_id: int, outcome: str, *, now: float | None = None) -> None:
        if outcome not in {"completed", "aborted", "failed"}:
            raise ValueError(f"Unsupported fully-async group outcome: {outcome}")
        now = monotonic() if now is None else now
        with self._lock:
            self._in_flight.pop(group_id, None)
            if outcome == "completed":
                self._completed_total += 1
                self._ready_at[group_id] = now
            elif outcome == "aborted":
                self._aborted_total += 1
            else:
                self._failed_total += 1

    def requeued(self, groups: int = 1) -> None:
        if groups < 1:
            raise ValueError("Successfully requeued group count must be positive")
        with self._lock:
            self._requeued_total += groups

    def delivered(self, group_ids: list[int], *, now: float | None = None) -> None:
        now = monotonic() if now is None else now
        with self._lock:
            for group_id in group_ids:
                # Delivery means the upstream worker has removed this completed
                # group from its queue and returned it to the training driver.
                self._ready_at.pop(group_id, None)
            self._delivered_total += len(group_ids)

    def observe_pending(self, pending_groups: int) -> None:
        if pending_groups < 0:
            raise ValueError("Completed pending group count must be nonnegative")
        with self._lock:
            self._pending_high_water = max(self._pending_high_water, pending_groups)

    def metrics(self, pending_groups: int, *, now: float | None = None) -> dict[str, float]:
        """Return one interval snapshot and advance interval counters.

        ``pending_groups`` comes directly from Slime's queue, while ready ages
        are recorded at the exact completion callback.  Python's ``qsize`` is
        still a concurrent snapshot, so this is deliberately an observability
        metric rather than a synchronization primitive.
        """
        if pending_groups < 0:
            raise ValueError("Completed pending group count must be nonnegative")
        now = monotonic() if now is None else now
        with self._lock:
            self._pending_high_water = max(self._pending_high_water, pending_groups)
            ready_ages = [max(0.0, now - ready_at) for ready_at in self._ready_at.values()]
            elapsed = max(1e-9, now - self._last_report_at)
            completed0, delivered0, requeued0 = self._last_report_counts
            metrics = {
                "rollout/async/groups/in_flight": float(len(self._in_flight)),
                "rollout/async/groups/completed_pending": float(pending_groups),
                "rollout/async/groups/completed_pending/high_water_since_last": float(
                    self._pending_high_water
                ),
                "rollout/async/groups/ready_age_seconds/mean": (
                    mean(ready_ages) if ready_ages else 0.0
                ),
                "rollout/async/groups/ready_age_seconds/max": max(ready_ages, default=0.0),
                "rollout/async/groups/started/total": float(self._started_total),
                "rollout/async/groups/completed/total": float(self._completed_total),
                "rollout/async/groups/delivered_to_train/total": float(self._delivered_total),
                "rollout/async/groups/aborted/total": float(self._aborted_total),
                "rollout/async/groups/requeued/total": float(self._requeued_total),
                "rollout/async/groups/failed/total": float(self._failed_total),
                "rollout/async/groups/completed_per_second": (
                    self._completed_total - completed0
                )
                / elapsed,
                "rollout/async/groups/delivered_to_train_per_second": (
                    self._delivered_total - delivered0
                )
                / elapsed,
                "rollout/async/groups/requeued_since_last": float(self._requeued_total - requeued0),
            }
            self._pending_high_water = pending_groups
            self._last_report_at = now
            self._last_report_counts = (
                self._completed_total,
                self._delivered_total,
                self._requeued_total,
            )
        return metrics


def _is_aborted_group(result: list[Any], sample_type: type | None) -> bool:
    status_type = getattr(sample_type, "Status", None)
    aborted = getattr(status_type, "ABORTED", None)
    return aborted is not None and any(getattr(sample, "status", None) == aborted for sample in result)


def _publish(metrics: dict[str, float]) -> None:
    # Nonblocking forwarding is important: observability must never make the
    # rollout producer wait for SwanLab or the logging actor.
    from .swanlab_bridge import report_metrics_nonblocking

    report_metrics_nonblocking(metrics)


class _ObservedDataBuffer:
    """Delegate Slime's buffer while counting only successful group requeues."""

    def __init__(self, delegate: Any, telemetry: AsyncGroupTelemetry):
        self._delegate = delegate
        self._telemetry = telemetry

    def get_samples(self, *args, **kwargs):
        return self._delegate.get_samples(*args, **kwargs)

    def add_samples(self, groups, *args, **kwargs):
        result = self._delegate.add_samples(groups, *args, **kwargs)
        self._telemetry.requeued(len(groups))
        return result

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


def install_fully_async_queue_metrics() -> bool:
    """Patch one imported Slime module; return false when Slime is unavailable.

    The launcher pins Slime to the commit whose ``AsyncRolloutWorker`` contract
    this wrapper observes.  No upstream file is edited or copied.
    """
    try:
        module = importlib.import_module("slime.rollout.fully_async_rollout")
    except ImportError:
        return False
    if getattr(module, _PATCH_MARKER, False):
        return True
    base_worker = getattr(module, "AsyncRolloutWorker", None)
    original_generate = getattr(module, "generate_rollout_fully_async", None)
    sample_type = getattr(module, "Sample", None)
    if not isinstance(base_worker, type) or not callable(original_generate):
        raise RuntimeError("Unsupported Slime fully-async rollout module")

    class ObservedAsyncRolloutWorker(base_worker):
        """Upstream worker with task and queue lifecycle counters added."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._noise_rl_group_telemetry = AsyncGroupTelemetry()
            # The upstream callback invokes ``self.data_buffer.add_samples``
            # for ABORTED groups.  Count requeues only after that call returns
            # successfully; a failed reinsertion is a distinct failure mode.
            self.data_buffer = _ObservedDataBuffer(self.data_buffer, self._noise_rl_group_telemetry)
            self._noise_rl_group_telemetry.observe_pending(self.queue_size())

        def _make_done_cb(self, group_id: int):
            # Upstream creates this callback immediately after it creates a
            # task, making it the narrowest stable hook for an actual launch.
            self._noise_rl_group_telemetry.started(group_id)
            upstream_callback = super()._make_done_cb(group_id)

            def observed_callback(done_task):
                outcome = "failed"
                try:
                    result = done_task.result()
                    if isinstance(result, list):
                        outcome = "aborted" if _is_aborted_group(result, sample_type) else "completed"
                except Exception:  # Upstream callback records the real exception.
                    pass
                try:
                    upstream_callback(done_task)
                finally:
                    self._noise_rl_group_telemetry.finished(group_id, outcome)
                    self._noise_rl_group_telemetry.observe_pending(self.queue_size())

            return observed_callback

        def get_completed_groups(self, limit: int | None = None):
            groups = super().get_completed_groups(limit)
            self._noise_rl_group_telemetry.delivered([group_id for group_id, _ in groups])
            self._noise_rl_group_telemetry.observe_pending(self.queue_size())
            return groups

    def observed_generate(args, rollout_id, data_buffer, evaluation: bool = False):
        started = monotonic()
        result = original_generate(args, rollout_id, data_buffer, evaluation=evaluation)
        worker = getattr(module, "_global_worker", None)
        telemetry = getattr(worker, "_noise_rl_group_telemetry", None)
        if telemetry is not None:
            metrics = telemetry.metrics(worker.queue_size())
            metrics.update(
                {
                    "rollout/async/rollout_id": float(rollout_id),
                    "rollout/async/target_groups": float(args.rollout_batch_size),
                    "rollout/async/collected_groups": float(len(result)),
                    "rollout/async/driver_wait_seconds": monotonic() - started,
                }
            )
            _publish(metrics)
        return result

    module.AsyncRolloutWorker = ObservedAsyncRolloutWorker
    module.generate_rollout_fully_async = observed_generate
    setattr(module, _PATCH_MARKER, True)
    logger.info("Installed project-local fully-async queue metrics wrapper")
    return True
