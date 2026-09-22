import json
from copy import deepcopy

from noise_rl.agent import Trajectory
from noise_rl.awm_model_replay import run_model_replay
from noise_rl.config import ExperimentConfig, NoiseConfig
from noise_rl.data import write_records
from noise_rl.fixtures import ByteTokenizer


def _record():
    return {
        "prompt": "Complete the tool-use task.",
        "metadata": {
            "task": {
                "id": "awm/payroll/0",
                "environment": "awm",
                "split": "train",
                "scenario": "payroll",
                "task_idx": 0,
                "read_only_tools": [],
            }
        },
    }


class _Client:
    def __init__(self, *_args):
        self.closed = False

    async def close(self):
        self.closed = True


async def _server_failure_trajectory(*_args):
    action = json.dumps({"tool_name": "create_schedule", "arguments": {"name": "Duplicate"}})
    trajectory = Trajectory(
        "prompt",
        [1],
        initial_observation='{"task":"Create a schedule","tools":[]}',
        environment="awm",
        termination="environment_terminal",
    )
    trajectory.steps = [
        {
            "turn": 0,
            "action": action,
            "format_error": False,
            "observation": "ENVIRONMENT_ERROR: server_error",
            "environment_info": {
                "awm": {
                    "tool_terminal_failure": True,
                    "tool_terminal_failure_type": "server_error",
                    "tool_terminal_failure_tool": "create_schedule",
                    "tool_terminal_failure_error": "HTTP 500",
                }
            },
        }
    ]
    trajectory.tool_calls = 1
    trajectory.elapsed_seconds = 0.25
    return trajectory


def test_model_replay_streams_trace_and_emits_non_excluding_candidates(tmp_path):
    manifest = tmp_path / "awm.jsonl"
    write_records(manifest, [_record()])
    output = tmp_path / "replay.jsonl"
    incidents = tmp_path / "incidents.jsonl"
    config = ExperimentConfig(noise=NoiseConfig())

    report = run_model_replay(
        manifest,
        output,
        config=config,
        model=None,
        url="http://sglang.invalid",
        awm_url="http://awm.invalid",
        repeats=2,
        concurrency=2,
        incident_output=incidents,
        tokenizer=ByteTokenizer(),
        client_factory=_Client,
        episode_runner=_server_failure_trajectory,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    candidate_rows = [json.loads(line) for line in incidents.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["status"] == "tool_terminal_failure" for row in rows)
    assert all(row["initial_observation"] == '{"task":"Create a schedule","tools":[]}' for row in rows)
    assert report["candidate_incident_count"] == 1
    assert report["terminal_tool_failure_counts"] == {"server_error": 2}
    assert candidate_rows == [
        {
            "arguments": {"name": "Duplicate"},
            "attempts": [0, 1],
            "error_examples": ["HTTP 500"],
            "evidence_output": str(output),
            "excluded_from_training": False,
            "failure_types": ["server_error"],
            "kind": "task_tool_incident",
            "observations": 2,
            "required_for_task": False,
            "review_note": "Candidate only: inspect the saved model trace and establish whether this exact action is required before setting required_for_task=true.",
            "scenario": "payroll",
            "source_kind": "model_rollout_candidate",
            "task_id": "awm/payroll/0",
            "task_idx": 0,
            "tool_name": "create_schedule",
        }
    ]


def test_model_replay_uses_stable_disjoint_shards():
    records = []
    for task_idx in range(64):
        record = deepcopy(_record())
        record["metadata"]["task"]["id"] = f"awm/payroll/{task_idx}"
        record["metadata"]["task"]["task_idx"] = task_idx
        records.append(record)
    # The public runner validates the selection; this regression test checks
    # that a stable manifest partition assigns each task to exactly one shard.
    from noise_rl.awm_model_replay import _selected_records

    left = _selected_records(records, shard_count=2, shard_index=0, tasks=0)
    right = _selected_records(records, shard_count=2, shard_index=1, tasks=0)
    left_ids = {row["metadata"]["task"]["id"] for row in left}
    right_ids = {row["metadata"]["task"]["id"] for row in right}
    assert not left_ids.intersection(right_ids)
    assert left_ids.union(right_ids) == {row["metadata"]["task"]["id"] for row in records}
