"""Build deterministic, scenario-disjoint manifests from local AWM data.

The OpenEnv AWM server indexes a task by ``(scenario, task_idx)``.  This
module reads the same local JSONL assets as that server, verifies that every
emitted task has a pure-code verifier, then partitions *whole scenarios* into
train and ``valid_unseen``.  It deliberately never imports OpenEnv or calls
Hugging Face, so manifest construction remains usable on an air-gapped host.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from .data import read_records, write_records


REQUIRED_AWM_FILES = (
    "gen_scenario.jsonl",
    "gen_tasks.jsonl",
    "gen_db.jsonl",
    "gen_sample.jsonl",
    "gen_envs.jsonl",
    "gen_verifier.jsonl",
    "gen_verifier.pure_code.jsonl",
)
DEFAULT_PROMPT = "Complete the tool-use task."


def normalize_scenario_name(scenario: str) -> str:
    """Match OpenEnv's scenario-name normalization exactly."""
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("Scenario name must be a nonempty string")
    value = re.sub(r"[^a-z0-9_]", "_", scenario.lower())
    return re.sub(r"_+", "_", value).strip("_")


def _read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing local AWM data file: {path}") from exc
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"AWM data file is empty: {path}")
    return rows


def _scenario_index(rows: list[dict], path: Path, field: str) -> dict[str, dict]:
    result = {}
    for line_number, row in enumerate(rows, 1):
        try:
            scenario = normalize_scenario_name(row[field])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid {field!r} in {path}:{line_number}") from exc
        if scenario in result:
            raise ValueError(f"Duplicate normalized scenario {scenario!r} in {path}")
        result[scenario] = row
    return result


def _task_index(rows: list[dict], path: Path) -> dict[str, list[str]]:
    result = {}
    for line_number, row in enumerate(rows, 1):
        try:
            scenario = normalize_scenario_name(row["scenario"])
            tasks = row["tasks"]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid task record in {path}:{line_number}") from exc
        if scenario in result:
            raise ValueError(f"Duplicate normalized scenario {scenario!r} in {path}")
        if not isinstance(tasks, list) or not tasks or not all(
            isinstance(task, str) and task.strip() for task in tasks
        ):
            raise ValueError(f"Expected a nonempty string task list in {path}:{line_number}")
        result[scenario] = tasks
    return result


def _pure_code_verifier_index(rows: list[dict], path: Path) -> dict[str, set[int]]:
    result = {}
    for line_number, row in enumerate(rows, 1):
        try:
            scenario = normalize_scenario_name(row["scenario"])
            task_idx = row["task_idx"]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid pure-code verifier record in {path}:{line_number}") from exc
        if type(task_idx) is not int or task_idx < 0:
            raise ValueError(f"Invalid task_idx in {path}:{line_number}")
        indices = result.setdefault(scenario, set())
        if task_idx in indices:
            raise ValueError(f"Duplicate code verifier for {scenario}/{task_idx} in {path}")
        indices.add(task_idx)
    return result


def _validate_sql_verifier_rows(rows: list[dict], path: Path) -> None:
    """Validate SQL-judge records without assuming a unique candidate per task.

    The upstream ``gen_verifier.jsonl`` may contain multiple LLM-judge
    candidates for one ``(scenario, task_idx)``.  OpenEnv returns the first
    matching candidate, while this project never uses SQL verification at all.
    Only the pure-code file has the one-verifier-per-task contract needed by
    the training harness.
    """
    for line_number, row in enumerate(rows, 1):
        try:
            normalize_scenario_name(row["scenario"])
            task_idx = row["task_idx"]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid SQL verifier record in {path}:{line_number}") from exc
        if type(task_idx) is not int or task_idx < 0:
            raise ValueError(f"Invalid task_idx in {path}:{line_number}")


def _load_catalog(data_root: str | Path) -> dict[str, list[str]]:
    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"AWM data root must be a directory: {root}")
    paths = {name: root / name for name in REQUIRED_AWM_FILES}
    rows = {name: _read_jsonl(path) for name, path in paths.items()}

    scenarios = _scenario_index(rows["gen_scenario.jsonl"], paths["gen_scenario.jsonl"], "name")
    tasks = _task_index(rows["gen_tasks.jsonl"], paths["gen_tasks.jsonl"])
    required_by_scenario = {
        "gen_tasks.jsonl": tasks,
        "gen_db.jsonl": _scenario_index(rows["gen_db.jsonl"], paths["gen_db.jsonl"], "scenario"),
        "gen_sample.jsonl": _scenario_index(rows["gen_sample.jsonl"], paths["gen_sample.jsonl"], "scenario"),
        "gen_envs.jsonl": _scenario_index(rows["gen_envs.jsonl"], paths["gen_envs.jsonl"], "scenario"),
    }
    scenario_names = set(scenarios)
    for filename, index in required_by_scenario.items():
        missing = sorted(scenario_names - set(index))
        unexpected = sorted(set(index) - scenario_names)
        if missing or unexpected:
            raise ValueError(
                f"Scenario mismatch for {filename}: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

    # The project uses only code verification, but validate the SQL verifier
    # file too: the local data directory should be complete enough for the
    # server started by train_awm_4gpu.sh.  SQL records are not unique by task.
    _validate_sql_verifier_rows(rows["gen_verifier.jsonl"], paths["gen_verifier.jsonl"])
    code_verifiers = _pure_code_verifier_index(
        rows["gen_verifier.pure_code.jsonl"], paths["gen_verifier.pure_code.jsonl"]
    )
    for scenario, task_list in tasks.items():
        expected = set(range(len(task_list)))
        actual = code_verifiers.get(scenario, set())
        if actual != expected:
            raise ValueError(
                f"Pure-code verifier coverage mismatch for {scenario}: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )
    return {scenario: tasks[scenario] for scenario in sorted(scenario_names)}


def _read_only_policy(path: str | Path | None, scenario_names: set[str]) -> tuple[list[str], dict[str, list[str]]]:
    if path is None:
        return [], {}
    source = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid read-only tool JSON: {source}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("Read-only tool file must be a JSON object")

    result = {}
    for raw_scenario, tools in value.items():
        if not isinstance(raw_scenario, str):
            raise ValueError("Read-only tool mapping keys must be strings")
        if not isinstance(tools, list) or not all(isinstance(name, str) and name for name in tools):
            raise ValueError(f"Read-only tools for {raw_scenario!r} must be a list of nonempty names")
        if len(set(tools)) != len(tools):
            raise ValueError(f"Read-only tools for {raw_scenario!r} contain duplicates")
        if raw_scenario == "_default":
            result[raw_scenario] = list(tools)
            continue
        scenario = normalize_scenario_name(raw_scenario)
        if scenario not in scenario_names:
            raise ValueError(f"Read-only mapping references unknown scenario: {raw_scenario!r}")
        result[scenario] = list(tools)
    return result.pop("_default", []), result


def _holdout_scenarios(scenarios: list[str], fraction: float, seed: int) -> set[str]:
    if not 0 < fraction < 1:
        raise ValueError("valid_scenario_fraction must be strictly between zero and one")
    if len(scenarios) < 2:
        raise ValueError("At least two scenarios are required for a scenario-disjoint split")
    count = math.ceil(len(scenarios) * fraction)
    count = min(max(count, 1), len(scenarios) - 1)

    def rank(scenario: str):
        digest = hashlib.sha256(f"awm-split-v1\0{seed}\0{scenario}".encode()).hexdigest()
        return digest, scenario

    return set(sorted(scenarios, key=rank)[:count])


def _manifest_rows(
    catalog: dict[str, list[str]],
    selected_scenarios: set[str],
    split: str,
    prompt: str,
    default_read_only_tools: list[str],
    scenario_read_only_tools: dict[str, list[str]],
) -> list[dict]:
    if split not in {"train", "valid_unseen"}:
        raise ValueError(f"Unsupported manifest split: {split}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    rows = []
    for scenario in sorted(selected_scenarios):
        for task_idx, _task_text in enumerate(catalog[scenario]):
            rows.append(
                {
                    "prompt": prompt,
                    "metadata": {
                        "task": {
                            "id": f"awm/{scenario}/{task_idx}",
                            "environment": "awm",
                            "split": split,
                            "scenario": scenario,
                            "task_idx": task_idx,
                            "read_only_tools": scenario_read_only_tools.get(
                                scenario, default_read_only_tools
                            ),
                        }
                    },
                }
            )
    if not rows:
        raise ValueError(f"No tasks selected for {split}")
    return rows


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_awm_manifests(
    data_root: str | Path,
    train_output: str | Path,
    valid_unseen_output: str | Path,
    *,
    valid_scenario_fraction: float = 0.2,
    seed: int = 20260910,
    read_only_tools: str | Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    report_output: str | Path | None = None,
) -> dict:
    """Build non-overlapping train and valid-unseen AWM manifests without downloads."""
    catalog = _load_catalog(data_root)
    train_output = Path(train_output).expanduser()
    valid_output = Path(valid_unseen_output).expanduser()
    report_path = Path(report_output).expanduser() if report_output else None
    outputs = [train_output, valid_output] + ([report_path] if report_path else [])
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Manifest and report output paths must be distinct")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")

    all_scenarios = sorted(catalog)
    valid_scenarios = _holdout_scenarios(all_scenarios, valid_scenario_fraction, seed)
    train_scenarios = set(all_scenarios) - valid_scenarios
    default_read_only, per_scenario_read_only = _read_only_policy(read_only_tools, set(all_scenarios))
    train_rows = _manifest_rows(
        catalog,
        train_scenarios,
        "train",
        prompt,
        default_read_only,
        per_scenario_read_only,
    )
    valid_rows = _manifest_rows(
        catalog,
        valid_scenarios,
        "valid_unseen",
        prompt,
        default_read_only,
        per_scenario_read_only,
    )

    # Run the same local schema validation used by the launcher before any
    # artifact is written.  It also guarantees globally unique task ids.
    for name, rows in (("train", train_rows), ("valid_unseen", valid_rows)):
        ids = [row["metadata"]["task"]["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate task id while building {name} manifest")
    train_output.parent.mkdir(parents=True, exist_ok=True)
    valid_output.parent.mkdir(parents=True, exist_ok=True)
    write_records(train_output, train_rows)
    try:
        write_records(valid_output, valid_rows)
    except BaseException:
        # The first output remains useful and is never overwritten; a failed
        # second write is explicit rather than silently replacing artifacts.
        raise
    read_records(train_output)
    read_records(valid_output)

    report = {
        "schema": 1,
        "dataset": "AgentWorldModel-1K",
        "data_root": str(Path(data_root).expanduser().resolve()),
        "source_files": {
            name: _file_sha256(Path(data_root).expanduser().resolve() / name)
            for name in REQUIRED_AWM_FILES
        },
        "seed": seed,
        "valid_scenario_fraction": valid_scenario_fraction,
        "prompt": prompt,
        "train": {
            "scenarios": sorted(train_scenarios),
            "scenario_count": len(train_scenarios),
            "task_count": len(train_rows),
            "output": str(train_output.resolve()),
        },
        "valid_unseen": {
            "scenarios": sorted(valid_scenarios),
            "scenario_count": len(valid_scenarios),
            "task_count": len(valid_rows),
            "output": str(valid_output.resolve()),
        },
        "read_only_tools": {
            "policy_file": str(Path(read_only_tools).expanduser().resolve()) if read_only_tools else None,
            "default": default_read_only,
            "explicit_scenario_count": len(per_scenario_read_only),
        },
    }
    if report_path:
        _write_report(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create scenario-disjoint local AWM train and valid-unseen manifests"
    )
    parser.add_argument("--data-root", required=True, help="Local AgentWorldModel-1K data directory")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--valid-unseen-output", required=True)
    parser.add_argument("--valid-scenario-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--read-only-tools",
        help="Optional JSON mapping of scenario to explicitly audited read-only tool names",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--report", help="Optional non-overwriting split audit JSON path")
    args = parser.parse_args(argv)
    report = build_awm_manifests(
        args.data_root,
        args.train_output,
        args.valid_unseen_output,
        valid_scenario_fraction=args.valid_scenario_fraction,
        seed=args.seed,
        read_only_tools=args.read_only_tools,
        prompt=args.prompt,
        report_output=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
