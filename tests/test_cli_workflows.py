import asyncio
import json
from types import SimpleNamespace

import pytest

from noise_rl import cli
from noise_rl.agent import Generation
from noise_rl.config import ExperimentConfig, NoiseConfig
from noise_rl.data import atomic_json, mini_records, write_records
from noise_rl.fixtures import ByteTokenizer


@pytest.fixture
def fixture_server(monkeypatch):
    transformers = pytest.importorskip("transformers")
    tokenizer = ByteTokenizer()
    tokenizer_calls = []

    def load_tokenizer(*args, **kwargs):
        tokenizer_calls.append((args, kwargs))
        return tokenizer

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)

    class Client:
        closed = 0
        local_tokenizer_calls = tokenizer_calls

        def __init__(self, *args):
            pass

        async def generate(self, tokens, params):
            transcript = tokenizer.decode(tokens)
            if "You clean apple 1." in transcript:
                action = "put apple 1 in/on table 1"
            elif "You take apple 1." in transcript:
                action = "clean apple 1 with sinkbasin 1"
            else:
                action = "take apple 1 from shelf 1"
            text = json.dumps({"action": action}) + "<|im_end|>"
            ids = tokenizer.encode(text)
            return Generation(ids, [-0.1] * len(ids), text, "stop")

        async def close(self):
            Client.closed += 1

    monkeypatch.setattr(cli, "SGLangClient", Client)
    return Client


def inputs(tmp_path):
    cfg = tmp_path / "config.json"
    data = tmp_path / "eval.jsonl"
    atomic_json(cfg, {"noise_rl": ExperimentConfig(noise=NoiseConfig(0, 0)).to_dict()})
    write_records(data, mini_records(1, "valid_unseen"))
    model = tmp_path / "model"
    model.mkdir()
    return SimpleNamespace(
        config=str(cfg),
        data=str(data),
        model=str(model),
        url="http://fixture",
        seed=123,
        output=str(tmp_path / "result.jsonl"),
        clean=False,
        repeats=2,
        temperature=0.8,
        scenarios=3,
        policy_samples=2,
        tasks=1,
    )


def test_standalone_evaluation_workflow(tmp_path, fixture_server):
    args = inputs(tmp_path)
    result = asyncio.run(cli.run_evaluation(args))
    assert result["episodes"] == 2 and result["success_rate_task_macro"] == 1
    rows = cli.load_episode_records(args.output)
    assert all(r["plan"]["evaluation"] for r in rows)
    assert len({r["plan"]["environment_seed"] for r in rows}) == 2
    assert fixture_server.closed == 1
    _, kwargs = fixture_server.local_tokenizer_calls[-1]
    assert kwargs["local_files_only"] is True
    assert kwargs["trust_remote_code"] is False
    with pytest.raises(FileExistsError):
        asyncio.run(cli.run_evaluation(args))


def test_frozen_policy_probe_workflow(tmp_path, fixture_server):
    args = inputs(tmp_path)
    report = asyncio.run(cli.run_probe(args))
    assert report["summary"]["episodes"] == 6
    matrix = next(iter(report["per_task"].values()))["rewards"]
    assert matrix == [[1, 1], [1, 1], [1, 1]]
    rows = cli.load_episode_records(args.output)
    assert len({r["plan"]["environment_seed"] for r in rows}) == 3
    assert len({r["plan"]["policy_seed"] for r in rows}) == 6
    assert all(r["phase"] == "frozen_policy_probe" for r in rows)
    assert fixture_server.closed == 1
    assert fixture_server.local_tokenizer_calls[-1][1]["local_files_only"] is True
