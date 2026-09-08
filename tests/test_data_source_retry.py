"""Unit coverage for the fully-async full-group retry contract."""

import copy
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

from noise_rl.config import ExperimentConfig
from noise_rl.data import NoiseDataSource, mini_records, write_records


class FakeSample:
    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    def __init__(self, **values):
        self.group_index = None
        self.index = None
        self.rollout_id = None
        self.prompt = ""
        self.metadata = {}
        self.status = self.Status.PENDING
        self.remove_sample = False
        for name, value in values.items():
            setattr(self, name, value)


@pytest.fixture
def fake_slime_sample(monkeypatch):
    slime = ModuleType("slime")
    slime.__path__ = []
    utils = ModuleType("slime.utils")
    utils.__path__ = []
    types = ModuleType("slime.utils.types")
    types.Sample = FakeSample
    slime.utils = utils
    utils.types = types
    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.utils", utils)
    monkeypatch.setitem(sys.modules, "slime.utils.types", types)
    return FakeSample


def _args(tmp_path, manifest):
    return SimpleNamespace(
        noise_rl=ExperimentConfig().to_dict(),
        n_samples_per_prompt=8,
        rollout_batch_size=2,
        over_sampling_batch_size=2,
        rollout_global_dataset=True,
        rollout_shuffle=False,
        prompt_data=str(manifest),
        save=str(tmp_path / "save"),
        load=None,
    )


def test_aborted_group_is_retried_first_and_cleaned(fake_slime_sample, tmp_path):
    manifest = tmp_path / "train.jsonl"
    write_records(manifest, mini_records(2, "train"))
    source = NoiseDataSource(_args(tmp_path, manifest))
    group = source.get_samples(1)[0]
    plans = [copy.deepcopy(sample.metadata["noise_plan"]) for sample in group]

    # A weight update can interrupt one member after others have completed.
    # Their old responses must never be mixed with the new policy's response.
    for sample in group:
        sample.status = fake_slime_sample.Status.COMPLETED
        sample.tokens = [1, 2, 3]
        sample.response = "old trajectory"
        sample.response_length = 2
        sample.reward = 1.0
        sample.loss_mask = [1, 1]
        sample.rollout_log_probs = [0.0, 0.0]
        sample.weight_versions = ["old-policy"]
        sample.metadata["noise_result"] = {"success": True}
    group[3].status = fake_slime_sample.Status.ABORTED

    source.add_samples([group])
    with pytest.raises(ValueError, match="already queued"):
        source.add_samples([group])

    retried, fresh = source.get_samples(2)
    assert retried is group
    assert source.counter == 2
    assert [sample.metadata["noise_plan"] for sample in retried] == plans
    assert all(sample.status == fake_slime_sample.Status.PENDING for sample in retried)
    assert all(sample.response == "" and sample.response_length == 0 for sample in retried)
    assert all(sample.reward is None and sample.tokens == [] for sample in retried)
    assert all("noise_result" not in sample.metadata for sample in retried)
    assert fresh[0].group_index == 1


def test_retry_rejects_partial_or_non_aborted_group(fake_slime_sample, tmp_path):
    manifest = tmp_path / "train.jsonl"
    write_records(manifest, mini_records(1, "train"))
    source = NoiseDataSource(_args(tmp_path, manifest))
    group = source.get_samples(1)[0]

    with pytest.raises(ValueError, match="Only a full group"):
        source.add_samples([group])
    with pytest.raises(ValueError, match="Partial/oversampled"):
        source.add_samples([group[:-1]])
