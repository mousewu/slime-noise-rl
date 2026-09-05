import os
import sys
from types import SimpleNamespace

import pytest

from noise_rl.config import ExperimentConfig
from noise_rl.data import mini_records

if os.environ.get("SLIME_SOURCE_PATH"):
    sys.path.insert(0, os.environ["SLIME_SOURCE_PATH"])


@pytest.fixture
def task():
    return mini_records(1, "train")[0]["metadata"]["task"]


@pytest.fixture
def slime_args(tmp_path):
    return SimpleNamespace(
        noise_rl=ExperimentConfig().to_dict(),
        n_samples_per_prompt=8,
        rollout_batch_size=2,
        over_sampling_batch_size=2,
        rollout_global_dataset=True,
        rollout_shuffle=True,
        save=str(tmp_path / "save"),
        load=None,
        advantage_estimator="grpo",
    )
