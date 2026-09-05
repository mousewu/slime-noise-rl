import copy
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

from .config import config_from_args
from .sampling import plan_sample, stable_seed


def atomic_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_records(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                task = record["metadata"]["task"]
                if not isinstance(task["id"], str) or not task["id"]:
                    raise ValueError("Task id must be nonempty")
                if task["environment"] not in {"mini", "alfworld"}:
                    raise ValueError("Unknown environment")
                if task["environment"] == "alfworld" and not Path(task["gamefile"]).is_absolute():
                    raise ValueError("ALFWorld gamefile must be an absolute path")
                records.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid manifest {path}:{line_number}: {exc}") from exc
    if not records or len({r["metadata"]["task"]["id"] for r in records}) != len(records):
        raise ValueError("Manifest must be nonempty and have unique task ids")
    return records


def write_records(path: str | Path, records: list[dict]):
    """Refuse to overwrite datasets; regenerating needs a new explicit output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def mini_records(count: int, split: str) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    return [
        {
            "prompt": "Complete the household task.",
            "metadata": {
                "task": {
                    "id": f"mini/{split}/{i:06d}",
                    "environment": "mini",
                    "split": split,
                    "item": f"apple {i + 1}",
                    "target": f"table {i + 1}",
                }
            },
        }
        for i in range(count)
    ]


def alfworld_records(root: str | Path, split: str, limit: int | None = None) -> list[dict]:
    if split not in {"train", "valid_seen", "valid_unseen"}:
        raise ValueError("Use official ALFWorld split names")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    root = Path(root).expanduser().resolve(strict=True)
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Expected {split_root}; root should be the json_2.1.1 directory")
    records = []
    for game in sorted(split_root.rglob("game.tw-pddl")):
        # Keep the exclusions and solvability filter used by the official loader.
        if "movable" in str(game) or "Sliced" in str(game):
            continue
        trajectory = game.with_name("traj_data.json")
        if not trajectory.is_file():
            continue
        raw = game.read_bytes()
        if not json.loads(raw).get("solvable", False):
            continue
        task_type = json.loads(trajectory.read_text(encoding="utf-8"))["task_type"]
        task = {
            "id": "alfworld/" + game.relative_to(root).parent.as_posix(),
            "environment": "alfworld",
            "split": split,
            "gamefile": str(game),
            "game_sha256": hashlib.sha256(raw).hexdigest(),
            "task_type": task_type,
        }
        records.append({"prompt": "Complete the household task.", "metadata": {"task": task}})
    if not records:
        raise ValueError(f"No solvable games found in {split_root}")
    if limit is not None:
        # Avoid a lexicographic prefix consisting of only one task family.
        return sorted(records, key=lambda r: stable_seed("manifest-subset", r["metadata"]["task"]["id"]))[
            :limit
        ]
    return records


class NoiseDataSource:
    """Slime DataSource protocol with exact groups and checkpointed task/seed counters."""

    def __init__(self, args):
        self.args = args
        self.config = config_from_args(args)
        self.records = read_records(args.prompt_data)
        for record in self.records:
            if record["metadata"]["task"].get("split") != "train":
                raise ValueError("Training manifest must use split=train; never train on evaluation tasks")
        encoded = json.dumps(self.records, sort_keys=True, ensure_ascii=True).encode()
        self.dataset_digest = hashlib.sha256(encoded).hexdigest()
        self.counter = 0
        self.metadata = {}
        self._epoch = None
        self._order = []

    def __len__(self):
        return len(self.records)

    def get_samples(self, num_samples):
        from slime.utils.types import Sample

        if type(num_samples) is not int or num_samples < 0:
            raise ValueError("num_samples must be a nonnegative integer")
        output = []
        for _ in range(num_samples):
            epoch, offset = divmod(self.counter, len(self.records))
            if epoch != self._epoch:
                self._order = list(range(len(self.records)))
                if getattr(self.args, "rollout_shuffle", False):
                    random.Random(stable_seed(self.config.seed, "task-order", epoch)).shuffle(self._order)
                self._epoch = epoch
            record = self.records[self._order[offset]]
            task_id = record["metadata"]["task"]["id"]
            group = []
            for rank in range(self.config.group_size):
                metadata = copy.deepcopy(record["metadata"])
                metadata["noise_plan"] = plan_sample(self.config, task_id, self.counter, rank).to_dict()
                index = self.counter * self.config.group_size + rank
                group.append(
                    Sample(
                        prompt=record.get("prompt", ""),
                        metadata=metadata,
                        group_index=self.counter,
                        index=index,
                        rollout_id=index,
                    )
                )
            output.append(group)
            self.counter += 1
        return output

    def add_samples(self, samples):
        if samples:
            raise RuntimeError("Partial/oversampled trajectory buffering is not supported in this experiment")

    def save(self, rollout_id):
        if not self.args.save:
            raise ValueError("Slime --save is required to checkpoint data-source state")
        atomic_json(
            Path(self.args.save) / "rollout" / f"noise_state_{rollout_id}.json",
            {
                "schema": 1,
                "counter": self.counter,
                "dataset_digest": self.dataset_digest,
                "config": self.config.to_dict(),
                "shuffle": bool(self.args.rollout_shuffle),
            },
        )

    def load(self, rollout_id=None):
        if rollout_id is None or rollout_id < 0 or not self.args.load:
            return
        path = Path(self.args.load) / "rollout" / f"noise_state_{rollout_id}.json"
        state = json.loads(path.read_text(encoding="utf-8"))  # Missing resume state is a hard error.
        expected = {
            "schema": 1,
            "dataset_digest": self.dataset_digest,
            "config": self.config.to_dict(),
            "shuffle": bool(self.args.rollout_shuffle),
        }
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("Resume dataset/config differs from the saved experiment")
        if type(state.get("counter")) is not int or state["counter"] < 0:
            raise ValueError("Invalid checkpoint counter")
        self.counter = state["counter"]
        self._epoch = None
