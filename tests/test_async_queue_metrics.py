import queue
import sys
from types import ModuleType, SimpleNamespace

from noise_rl import async_queue_metrics
from noise_rl.async_queue_metrics import AsyncGroupTelemetry


def test_async_group_telemetry_tracks_pending_age_through_delivery():
    telemetry = AsyncGroupTelemetry()
    telemetry.started(4, now=10.0)
    telemetry.finished(4, "completed", now=12.0)
    telemetry.observe_pending(1)

    metrics = telemetry.metrics(1, now=15.0)
    assert metrics["rollout/async/groups/in_flight"] == 0
    assert metrics["rollout/async/groups/completed_pending"] == 1
    assert metrics["rollout/async/groups/completed_pending/high_water_since_last"] == 1
    assert metrics["rollout/async/groups/ready_age_seconds/max"] == 3
    assert metrics["rollout/async/groups/completed/total"] == 1

    telemetry.started(5, now=15.5)
    telemetry.finished(5, "aborted", now=16.0)
    telemetry.requeued()
    telemetry.delivered([4], now=16.0)
    metrics = telemetry.metrics(0, now=17.0)
    assert metrics["rollout/async/groups/completed_pending"] == 0
    assert metrics["rollout/async/groups/ready_age_seconds/max"] == 0
    assert metrics["rollout/async/groups/delivered_to_train/total"] == 1
    assert metrics["rollout/async/groups/aborted/total"] == 1
    assert metrics["rollout/async/groups/requeued/total"] == 1


def test_runtime_wrapper_observes_upstream_queue_without_editing_slime(monkeypatch):
    slime = ModuleType("slime")
    rollout = ModuleType("slime.rollout")
    upstream = ModuleType("slime.rollout.fully_async_rollout")

    class Sample:
        class Status:
            ABORTED = "aborted"

    class BaseWorker:
        def __init__(self, *_args, **_kwargs):
            self.output_queue = queue.Queue()
            self.data_buffer = SimpleNamespace(
                get_samples=lambda *_args, **_kwargs: [], add_samples=lambda *_args, **_kwargs: None
            )

        def queue_size(self):
            return self.output_queue.qsize()

        def _make_done_cb(self, group_id):
            def callback(task):
                result = task.result()
                if not any(sample.status == Sample.Status.ABORTED for sample in result):
                    self.output_queue.put((group_id, result))

            return callback

        def get_completed_groups(self, limit=None):
            result = []
            while limit is None or len(result) < limit:
                try:
                    result.append(self.output_queue.get_nowait())
                except queue.Empty:
                    break
            return result

    upstream.Sample = Sample
    upstream.AsyncRolloutWorker = BaseWorker
    upstream._global_worker = None

    def generate(args, _rollout_id, _data_buffer, evaluation=False):
        assert not evaluation
        worker = upstream._global_worker
        return [group for _, group in worker.get_completed_groups(args.rollout_batch_size)]

    upstream.generate_rollout_fully_async = generate
    slime.rollout = rollout
    rollout.fully_async_rollout = upstream
    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.rollout", rollout)
    monkeypatch.setitem(sys.modules, "slime.rollout.fully_async_rollout", upstream)
    published = []
    monkeypatch.setattr(async_queue_metrics, "_publish", lambda metrics: published.append(metrics))

    assert async_queue_metrics.install_fully_async_queue_metrics()
    worker = upstream.AsyncRolloutWorker()
    upstream._global_worker = worker
    sample = SimpleNamespace(status="completed")
    callback = worker._make_done_cb(7)
    callback(SimpleNamespace(result=lambda: [sample]))

    args = SimpleNamespace(rollout_batch_size=1)
    result = upstream.generate_rollout_fully_async(args, 3, object())

    assert result == [[sample]]
    assert published[-1]["rollout/async/rollout_id"] == 3
    assert published[-1]["rollout/async/groups/delivered_to_train/total"] == 1
    assert published[-1]["rollout/async/groups/completed_pending"] == 0
