import json
import logging
import sys
from types import ModuleType, SimpleNamespace

from noise_rl import swanlab_bridge
from noise_rl.swanlab_bridge import (
    SwanLabLogger,
    _BoundedRayForwarder,
    _SwanLabLogMirror,
    _RolloutTimingBuffer,
    install_slime_logging_patch,
    report_rollout_timing,
    scalar_metrics,
)


class ItemScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def test_scalar_metrics_detaches_framework_values_and_rejects_nonfinite():
    assert scalar_metrics(
        {
            "loss": ItemScalar(0.25),
            "step": 2,
            "enabled": True,
            "nan": float("nan"),
            "text": "skip",
            "vector": ItemScalar([1, 2]),
        }
    ) == {"loss": 0.25, "step": 2, "enabled": 1}


def test_logger_leaves_global_step_to_sdk_and_keeps_slime_step(monkeypatch, tmp_path):
    calls = []
    fake = ModuleType("swanlab")
    fake.init = lambda **kwargs: calls.append(("init", kwargs)) or SimpleNamespace(id=kwargs["id"])
    fake.log = lambda values: calls.append(("log", values))
    fake.finish = lambda: calls.append(("finish",))
    monkeypatch.setitem(sys.modules, "swanlab", fake)

    logger = SwanLabLogger(
        {
            "project": "research",
            "experiment_name": "matched-s42",
            "mode": "offline",
            "logdir": str(tmp_path / "swanlab"),
            "id": "run-id",
            "resume": "allow",
            "config": {"seed": 42},
        }
    )
    assert logger.ready() == {"id": "run-id"}
    logger.log({"train/step": 7, "train/loss": 0.5})
    logger.log({"rollout/step": 2, "rollout/reward": 0.75})
    logger.finish()

    assert calls[0][0] == "init" and calls[0][1]["config"] == {"seed": 42}
    assert calls[1] == (
        "log",
        {"train/step": 7, "train/loss": 0.5},
    )
    assert calls[2] == (
        "log",
        {"rollout/step": 2, "rollout/reward": 0.75},
    )
    assert calls[3] == ("finish",)
    events = [json.loads(line) for line in (tmp_path / "swanlab" / "metric_events.jsonl").read_text().splitlines()]
    assert [event["metrics"] for event in events] == [calls[1][1], calls[2][1]]


def test_logger_mirrors_worker_text_to_swanlab_stdout_and_audit(monkeypatch, tmp_path, capsys):
    fake = ModuleType("swanlab")
    fake.init = lambda **kwargs: SimpleNamespace(id=kwargs["id"])
    fake.log = lambda values: None
    fake.finish = lambda: None
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    monkeypatch.setenv("SWANLAB_API_KEY", "secret-value")

    logger = SwanLabLogger(
        {
            "project": "research",
            "experiment_name": "matched-s42",
            "mode": "offline",
            "logdir": str(tmp_path / "swanlab"),
            "id": "run-id",
            "resume": "allow",
        }
    )
    logger.log_text(
        {"time": 1.0, "level": "INFO", "logger": "DetailLogger", "message": "key=secret-value"}
    )

    assert "[noise-rl][INFO][DetailLogger] key=<SWANLAB_API_KEY_REDACTED>" in capsys.readouterr().out
    events = [json.loads(line) for line in (tmp_path / "swanlab" / "forwarded_logs.jsonl").read_text().splitlines()]
    assert events == [
        {
            "event": 1,
            "level": "INFO",
            "logger": "DetailLogger",
            "message": "key=<SWANLAB_API_KEY_REDACTED>",
            "time": 1.0,
        }
    ]


def test_worker_log_handler_submits_nonblocking_text(monkeypatch):
    forwarded = []

    class RemoteLogText:
        def remote(self, payload):
            forwarded.append(payload)

    monkeypatch.setattr(
        swanlab_bridge,
        "_logger_actor",
        lambda: SimpleNamespace(log_text=RemoteLogText()),
    )
    monkeypatch.setenv(swanlab_bridge.SWANLAB_ACTOR_ENV, "logger")
    handler = _SwanLabLogMirror()
    record = logging.LogRecord("DetailLogger", logging.WARNING, __file__, 1, "slow rollout %s", (7,), None)

    handler.emit(record)

    assert forwarded[0]["level"] == "WARNING"
    assert forwarded[0]["logger"] == "DetailLogger"
    assert forwarded[0]["message"] == "slow rollout 7"


def test_bounded_forwarder_drops_new_logs_when_actor_is_not_draining(monkeypatch):
    references = []
    forwarder = _BoundedRayForwarder(limit_environment="TEST_MAX_PENDING", default_limit=2)
    monkeypatch.setenv("TEST_MAX_PENDING", "2")

    assert forwarder.submit(lambda: references.append(object()) or references[-1]) == "submitted"
    assert forwarder.submit(lambda: references.append(object()) or references[-1]) == "submitted"
    assert forwarder.submit(lambda: references.append(object()) or references[-1]) == "dropped"
    assert len(references) == 2
    assert forwarder.snapshot() == {
        "pending": 2,
        "submitted_total": 2,
        "dropped_total": 1,
        "failed_total": 0,
    }


def test_rollout_timing_buffer_emits_windowed_environment_metrics():
    buffer = _RolloutTimingBuffer()
    first = {
        "environment_runner_wait_seconds": 1.0,
        "environment_queue_seconds": 2.0,
        "environment_step_seconds": 3.0,
        "model_request_seconds": 4.0,
        "elapsed_seconds": 5.0,
    }
    second = {
        "environment_runner_wait_seconds": 3.0,
        "environment_queue_seconds": 4.0,
        "environment_step_seconds": 5.0,
        "model_request_seconds": 6.0,
        "elapsed_seconds": 7.0,
    }

    assert buffer.add(first, group_id=4, window_size=2) is None
    metrics = buffer.add(second, group_id=5, window_size=2)

    assert metrics == {
        "rollout/stream/episodes": 2.0,
        "rollout/stream/group_id/max": 5.0,
        "rollout/stream/environment_runner_wait_seconds/mean": 2.0,
        "rollout/stream/environment_runner_wait_seconds/max": 3.0,
        "rollout/stream/environment_queue_seconds/mean": 3.0,
        "rollout/stream/environment_queue_seconds/max": 4.0,
        "rollout/stream/environment_step_seconds/mean": 4.0,
        "rollout/stream/environment_step_seconds/max": 5.0,
        "rollout/stream/model_request_seconds/mean": 5.0,
        "rollout/stream/model_request_seconds/max": 6.0,
        "rollout/stream/elapsed_seconds/mean": 6.0,
        "rollout/stream/elapsed_seconds/max": 7.0,
    }


def test_rollout_timing_forwards_nonblocking_after_one_group(monkeypatch):
    forwarded = []

    class RemoteLog:
        def remote(self, metrics):
            forwarded.append(metrics)

    monkeypatch.setattr(
        swanlab_bridge,
        "_logger_actor",
        lambda: SimpleNamespace(log=RemoteLog()),
    )
    monkeypatch.setattr(swanlab_bridge, "_ROLLOUT_TIMING_BUFFER", _RolloutTimingBuffer())
    monkeypatch.setenv(swanlab_bridge.SWANLAB_ACTOR_ENV, "logger")
    record = {
        "environment_runner_wait_seconds": 0.25,
        "environment_queue_seconds": 0.5,
        "environment_step_seconds": 1.0,
        "model_request_seconds": 2.0,
        "elapsed_seconds": 3.0,
    }

    assert not report_rollout_timing(record, group_id=8, group_size=2)
    assert report_rollout_timing(record, group_id=8, group_size=2)
    assert len(forwarded) == 1
    assert {
        key: value for key, value in forwarded[0].items() if not key.startswith("swanlab/bridge/")
    } == {
        "rollout/stream/episodes": 2.0,
        "rollout/stream/group_id/max": 8.0,
        "rollout/stream/environment_runner_wait_seconds/mean": 0.25,
        "rollout/stream/environment_runner_wait_seconds/max": 0.25,
        "rollout/stream/environment_queue_seconds/mean": 0.5,
        "rollout/stream/environment_queue_seconds/max": 0.5,
        "rollout/stream/environment_step_seconds/mean": 1.0,
        "rollout/stream/environment_step_seconds/max": 1.0,
        "rollout/stream/model_request_seconds/mean": 2.0,
        "rollout/stream/model_request_seconds/max": 2.0,
        "rollout/stream/elapsed_seconds/mean": 3.0,
        "rollout/stream/elapsed_seconds/max": 3.0,
    }


def test_runtime_patch_preserves_slime_logger_and_forwards_once(monkeypatch):
    original_calls = []
    forwarded = []

    logging_utils = ModuleType("slime.observability.logging_utils")
    logging_utils.log = lambda args, metrics, step_key: original_calls.append((args, metrics, step_key))
    slime = ModuleType("slime")
    observability = ModuleType("slime.observability")
    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.observability", observability)
    monkeypatch.setitem(sys.modules, "slime.observability.logging_utils", logging_utils)

    class RemoteLog:
        def remote(self, metrics):
            forwarded.append(metrics)
            return "done"

    handle = SimpleNamespace(log=RemoteLog())
    ray = ModuleType("ray")
    ray.get_actor = lambda name: handle
    ray.get = lambda value, timeout=None: value
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setenv(swanlab_bridge.SWANLAB_ACTOR_ENV, "logger")
    monkeypatch.setattr(swanlab_bridge, "_LOGGER_ACTOR", None)
    monkeypatch.setattr(swanlab_bridge, "_FORWARD_WARNING_EMITTED", False)

    install_slime_logging_patch()
    install_slime_logging_patch()
    args = object()
    metrics = {"train/step": 3, "train/loss": ItemScalar(0.125)}
    logging_utils.log(args, metrics, "train/step")

    assert original_calls == [(args, metrics, "train/step")]
    assert {key: value for key, value in forwarded[0].items() if not key.startswith("swanlab/bridge/")} == {
        "train/step": 3,
        "train/loss": 0.125,
    }
