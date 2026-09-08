import copy
import hashlib
import json
import os
import random
import tempfile
from collections import deque
from pathlib import Path

from .config import config_from_args
from .sampling import SamplingPlan, plan_sample, stable_seed


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
                if task["environment"] not in {"mini", "alfworld", "awm"}:
                    raise ValueError("Unknown environment")
                if task["environment"] == "awm":
                    if not isinstance(task.get("scenario"), str) or not task["scenario"]:
                        raise ValueError("AWM requires scenario")
                    if type(task.get("task_idx")) is not int or task["task_idx"] < 0:
                        raise ValueError("AWM requires nonnegative task_idx")
                    if not isinstance(task.get("read_only_tools", []), list) or not all(isinstance(t, str) for t in task.get("read_only_tools", [])):
                        raise ValueError("read_only_tools must be a list of names")
                if task["environment"] == "alfworld" and not Path(task["gamefile"]).is_absolute():
                    raise ValueError("ALFWorld gamefile must be an absolute path")
                records.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid manifest {path}:{line_number}: {exc}") from exc
    if not records or len({r["metadata"]["task"]["id"] for r in records}) != len(records):
        raise ValueError("Manifest must be nonempty and have unique task ids")
    return records


def validate_local_records(records: list[dict]):
    """Verify every environment asset referenced by a manifest is local and unchanged."""
    checked = set()
    for record in records:
        task = record["metadata"]["task"]
        if task["environment"] != "alfworld":
            continue
        game = Path(task["gamefile"]).expanduser()
        if not game.is_file():
            raise FileNotFoundError(f"Manifest references a missing local ALFWorld game: {game}")
        resolved = game.resolve()
        if resolved in checked:
            continue
        expected = task.get("game_sha256")
        if expected and hashlib.sha256(resolved.read_bytes()).hexdigest() != expected:
            raise ValueError(f"ALFWorld game changed since manifest creation: {resolved}")
        checked.add(resolved)


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
    """Slime data source with exact groups and a FIFO retry queue.

    A fully-async Slime worker sends an entire group back through
    :meth:`add_samples` when one of its members is interrupted for a weight
    update.  Retrying only the interrupted member would break the matched
    comparison and LOO baseline, so this source accepts *only* an untouched
    complete group and returns it before allocating any new task.
    """

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
        self._retry_buffer = deque()
        self._queued_group_ids = set()

    def __len__(self):
        return len(self.records)

    def get_samples(self, num_samples):
        from slime.utils.types import Sample

        if type(num_samples) is not int or num_samples < 0:
            raise ValueError("num_samples must be a nonnegative integer")
        output = self._take_retry_groups(num_samples, Sample)
        for _ in range(num_samples - len(output)):
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
        """Queue a full aborted group for a clean, same-identity retry.

        Slime's fully-async rollout worker calls this after it observes an
        ``ABORTED`` member.  We deliberately do not support partial-rollout
        buffering or arbitrary oversampling: either case would change the
        members of a matched comparison group.
        """
        from slime.utils.types import Sample

        if not samples:
            return
        if type(samples) is not list:
            raise TypeError("NoiseDataSource.add_samples expects a list of complete groups")

        group_ids = []
        # Validate the complete input before mutating the FIFO, so an invalid
        # second group cannot leave the first one partially accepted.
        for group in samples:
            group_ids.append(self._validate_retry_group(group, Sample))
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("Cannot queue the same comparison group more than once")
        duplicate_ids = set(group_ids) & self._queued_group_ids
        if duplicate_ids:
            raise ValueError(f"Comparison group(s) already queued for retry: {sorted(duplicate_ids)}")

        self._retry_buffer.extend(samples)
        self._queued_group_ids.update(group_ids)

    def _take_retry_groups(self, num_samples, Sample):
        output = []
        while self._retry_buffer and len(output) < num_samples:
            group = self._retry_buffer.popleft()
            group_id = group[0].group_index
            self._queued_group_ids.remove(group_id)
            self._reset_group_for_clean_retry(group, Sample)
            output.append(group)
        return output

    def _validate_retry_group(self, group, Sample):
        if type(group) is not list:
            raise TypeError("NoiseDataSource only buffers complete list[Sample] groups")
        if len(group) != self.config.group_size:
            raise ValueError(
                "Partial/oversampled trajectory buffering is not supported: "
                f"expected {self.config.group_size} samples, got {len(group)}"
            )
        if not all(isinstance(sample, Sample) for sample in group):
            raise TypeError("NoiseDataSource only buffers Slime Sample instances")

        group_id = group[0].group_index
        if type(group_id) is not int or group_id < 0:
            raise ValueError("Buffered comparison group requires a nonnegative integer group_index")
        task_id = None
        for rank, sample in enumerate(group):
            if sample.group_index != group_id:
                raise ValueError("Buffered samples must all belong to one comparison group")
            expected_index = group_id * self.config.group_size + rank
            if sample.index != expected_index or sample.rollout_id != expected_index:
                raise ValueError("Buffered samples must retain their original ordered index and rollout_id")
            if sample.remove_sample:
                raise ValueError("Cannot retry a comparison group with removed members")
            metadata = sample.metadata
            if not isinstance(metadata, dict) or not isinstance(metadata.get("task"), dict):
                raise ValueError("Buffered samples require task and noise-plan metadata")
            current_task_id = metadata["task"].get("id")
            raw_plan = metadata.get("noise_plan")
            if not isinstance(current_task_id, str) or not current_task_id or not isinstance(raw_plan, dict):
                raise ValueError("Buffered samples require a task id and serialized noise plan")
            try:
                plan = SamplingPlan(**raw_plan)
            except TypeError as exc:
                raise ValueError("Buffered samples have an invalid noise plan") from exc
            if task_id is None:
                task_id = current_task_id
            if current_task_id != task_id or plan.task_id != task_id:
                raise ValueError("Buffered comparison groups must use one task id")
            expected_plan = plan_sample(self.config, task_id, group_id, rank)
            if plan != expected_plan:
                raise ValueError("Buffered samples must retain the original training noise plan")

        if not any(sample.status == Sample.Status.ABORTED for sample in group):
            raise ValueError("Only a full group containing an ABORTED member may be retried")
        return group_id

    @staticmethod
    def _reset_group_for_clean_retry(group, Sample):
        """Clear old rollout artifacts so every member is regenerated together.

        ``generate_and_rm`` normally skips COMPLETED/TRUNCATED samples.  A
        group interrupted halfway through therefore must not be handed back as
        is: otherwise it would combine trajectories from two policy versions.
        """
        for sample in group:
            sample.tokens = []
            sample.response = ""
            sample.response_length = 0
            sample.reward = None
            sample.loss_mask = []
            sample.rollout_log_probs = []
            sample.rollout_top_p_token_ids = None
            sample.rollout_top_p_token_offsets = None
            sample.rollout_routed_experts = None
            sample.weight_versions = []
            sample.teacher_log_probs = None
            sample.non_generation_time = 0.0
            if hasattr(sample, "spec_info"):
                sample.spec_info = type(sample.spec_info)()
            if hasattr(sample, "prefix_cache_info"):
                sample.prefix_cache_info = type(sample.prefix_cache_info)()
            sample.metadata.pop("noise_result", None)
            sample.status = Sample.Status.PENDING

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
