import sys
import json
from types import ModuleType, SimpleNamespace

from noise_rl import swanlab_bridge
from noise_rl.swanlab_bridge import SwanLabLogger, install_slime_logging_patch, scalar_metrics


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
    assert forwarded == [{"train/step": 3, "train/loss": 0.125}]
