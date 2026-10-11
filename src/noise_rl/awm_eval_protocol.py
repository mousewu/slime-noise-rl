"""Build leakage-audited AWM manifests for post-training checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .data import atomic_json, read_records, write_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _task(record: dict[str, Any]) -> dict[str, Any]:
    task = record["metadata"]["task"]
    if task.get("environment") != "awm" or task.get("split") != "train":
        raise ValueError("AWM evaluation protocol accepts only train-split AWM manifests")
    return task


def build_awm_eval_protocol(
    environment_manifest: str | Path,
    rl_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create three auditable sets plus their deduplicated replay union.

    ``environment_non_rl`` tests task-level generalization. ``scenario_heldout``
    is its stricter subset whose scenario never occurs in RL. ``rl_resample``
    contains the trained tasks and is evaluated with fresh policy seeds by the
    replay runner. The union avoids evaluating the held-out subset twice.
    """
    environment_path = Path(environment_manifest).expanduser().resolve(strict=True)
    rl_path = Path(rl_manifest).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation protocol: {destination}")

    environment = read_records(environment_path)
    rl = read_records(rl_path)
    environment_by_id = {_task(row)["id"]: row for row in environment}
    rl_by_id = {_task(row)["id"]: row for row in rl}
    missing = sorted(set(rl_by_id) - set(environment_by_id))
    if missing:
        raise ValueError(f"RL-certified manifest is not a subset of environment-certified: {missing[0]}")

    rl_ids = set(rl_by_id)
    rl_scenarios = {_task(row)["scenario"] for row in rl}
    environment_non_rl = [row for row in environment if _task(row)["id"] not in rl_ids]
    scenario_heldout = [
        row for row in environment_non_rl if _task(row)["scenario"] not in rl_scenarios
    ]
    if not environment_non_rl:
        raise ValueError("No environment-certified task remains outside the RL training manifest")
    if not scenario_heldout:
        raise ValueError("No scenario-held-out task remains; create a scenario-disjoint split before training")

    destination.mkdir(parents=True)
    paths = {
        "environment_non_rl": destination / "environment_non_rl.jsonl",
        "scenario_heldout": destination / "scenario_heldout.jsonl",
        "rl_resample": destination / "rl_resample.jsonl",
        "replay_union": destination / "replay_union.jsonl",
    }
    write_records(paths["environment_non_rl"], environment_non_rl)
    write_records(paths["scenario_heldout"], scenario_heldout)
    write_records(paths["rl_resample"], rl)
    # scenario_heldout is already a subset of environment_non_rl.
    replay_union = environment_non_rl + rl
    write_records(paths["replay_union"], replay_union)

    report = {
        "schema": 1,
        "phase": "awm_checkpoint_eval_protocol",
        "policy": {
            "environment_non_rl": "environment-certified tasks absent from RL training",
            "scenario_heldout": "environment-certified tasks whose complete scenario is absent from RL training",
            "rl_resample": "RL training tasks, evaluated only with fresh model-sampling seeds and fresh AWM sessions",
            "replay_union": "deduplicated union evaluated once per checkpoint",
        },
        "source": {
            "environment_manifest": str(environment_path),
            "environment_manifest_sha256": _sha256(environment_path),
            "rl_manifest": str(rl_path),
            "rl_manifest_sha256": _sha256(rl_path),
        },
        "counts": {
            "environment_certified": len(environment),
            "rl_certified": len(rl),
            "environment_non_rl": len(environment_non_rl),
            "scenario_heldout": len(scenario_heldout),
            "rl_resample": len(rl),
            "replay_union": len(replay_union),
            "rl_scenarios": len(rl_scenarios),
            "scenario_heldout_scenarios": len({_task(row)["scenario"] for row in scenario_heldout}),
        },
        "sets": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "task_ids": [_task(row)["id"] for row in (
                    environment_non_rl if name == "environment_non_rl" else
                    scenario_heldout if name == "scenario_heldout" else
                    rl if name == "rl_resample" else replay_union
                )],
            }
            for name, path in paths.items()
        },
    }
    atomic_json(destination / "protocol.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-manifest", required=True, type=Path)
    parser.add_argument("--rl-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_awm_eval_protocol(
        args.environment_manifest, args.rl_manifest, args.output_dir
    )
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
