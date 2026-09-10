import json

import pytest

from noise_rl.awm_manifest import build_awm_manifests, normalize_scenario_name
from noise_rl.data import read_records


REQUIRED = (
    "gen_scenario.jsonl",
    "gen_tasks.jsonl",
    "gen_db.jsonl",
    "gen_sample.jsonl",
    "gen_envs.jsonl",
    "gen_verifier.jsonl",
    "gen_verifier.pure_code.jsonl",
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_awm_data(tmp_path):
    root = tmp_path / "AgentWorldModel-1K"
    root.mkdir()
    scenarios = ["E-Commerce 33", "Library 2", "Travel 1", "Warehouse 9"]
    write_jsonl(root / "gen_scenario.jsonl", [{"name": name} for name in scenarios])
    write_jsonl(
        root / "gen_tasks.jsonl",
        [{"scenario": name, "tasks": [f"{name} task 0", f"{name} task 1"]} for name in scenarios],
    )
    for filename in ("gen_db.jsonl", "gen_sample.jsonl", "gen_envs.jsonl"):
        write_jsonl(root / filename, [{"scenario": name} for name in scenarios])
    verifier_rows = [
        {"scenario": name, "task_idx": index, "verifier": "pass"}
        for name in scenarios
        for index in range(2)
    ]
    write_jsonl(root / "gen_verifier.jsonl", verifier_rows)
    write_jsonl(root / "gen_verifier.pure_code.jsonl", verifier_rows)
    assert all((root / filename).is_file() for filename in REQUIRED)
    return root


def test_normalization_matches_openenv_contract():
    assert normalize_scenario_name(" E-Commerce__33 ") == "e_commerce_33"


def test_manifest_builder_partitions_complete_scenarios_and_records_audit(tmp_path):
    root = make_awm_data(tmp_path)
    policy = tmp_path / "read-only.json"
    policy.write_text(
        json.dumps({"_default": ["inspect"], "E-Commerce 33": ["search_products"]}),
        encoding="utf-8",
    )
    train_path = tmp_path / "manifests" / "train.jsonl"
    valid_path = tmp_path / "manifests" / "valid_unseen.jsonl"
    report_path = tmp_path / "manifests" / "split.json"

    report = build_awm_manifests(
        root,
        train_path,
        valid_path,
        valid_scenario_fraction=0.5,
        seed=7,
        read_only_tools=policy,
        report_output=report_path,
    )

    train, valid = read_records(train_path), read_records(valid_path)
    train_scenarios = {row["metadata"]["task"]["scenario"] for row in train}
    valid_scenarios = {row["metadata"]["task"]["scenario"] for row in valid}
    assert len(train) == len(valid) == 4
    assert train_scenarios.isdisjoint(valid_scenarios)
    assert train_scenarios | valid_scenarios == {
        "e_commerce_33",
        "library_2",
        "travel_1",
        "warehouse_9",
    }
    for row in train + valid:
        task = row["metadata"]["task"]
        assert task["id"] == f"awm/{task['scenario']}/{task['task_idx']}"
        expected = ["search_products"] if task["scenario"] == "e_commerce_33" else ["inspect"]
        assert task["read_only_tools"] == expected
    assert report["train"]["task_count"] == 4
    assert json.loads(report_path.read_text(encoding="utf-8"))["valid_unseen"]["task_count"] == 4


def test_manifest_builder_rejects_missing_pure_code_verifier_before_writing(tmp_path):
    root = make_awm_data(tmp_path)
    verifier = root / "gen_verifier.pure_code.jsonl"
    rows = [json.loads(line) for line in verifier.read_text(encoding="utf-8").splitlines()]
    write_jsonl(verifier, rows[:-1])

    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    with pytest.raises(ValueError, match="Pure-code verifier coverage mismatch"):
        build_awm_manifests(root, train_path, valid_path, valid_scenario_fraction=0.5)
    assert not train_path.exists() and not valid_path.exists()


def test_manifest_builder_never_overwrites_existing_outputs(tmp_path):
    root = make_awm_data(tmp_path)
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    build_awm_manifests(root, train_path, valid_path, valid_scenario_fraction=0.5)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_awm_manifests(root, train_path, valid_path, valid_scenario_fraction=0.5)
