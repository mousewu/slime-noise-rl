"""Runs against actual upstream Sample implementation, not a replacement dataclass."""

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from noise_rl import slime_hooks
from noise_rl.agent import QwenChatProtocol, SGLangAbort, run_episode
from noise_rl.config import ExperimentConfig, NoiseConfig
from noise_rl.data import NoiseDataSource, mini_records, write_records
from noise_rl.fixtures import ByteTokenizer, ScriptedClient
from noise_rl.launch import model_arguments, verify_slime
from noise_rl.sampling import plan_sample
from noise_rl.slime_hooks import fill_sample, reward_postprocess, validate_slime_args

pytestmark = pytest.mark.integration


@pytest.fixture
def Sample():
    return pytest.importorskip("slime.utils.types", reason="Install Slime or set SLIME_SOURCE_PATH").Sample


def test_actual_slime_sample_token_alignment(Sample, task, slime_args):
    cfg = ExperimentConfig(noise=NoiseConfig(0, 0))
    plan = plan_sample(cfg, task["id"], 0, 0)
    trajectory = asyncio.run(run_episode(task, plan, cfg, ByteTokenizer(), ScriptedClient(task)))
    sample = fill_sample(slime_args, Sample(), trajectory)
    assert sample.tokens == trajectory.tokens
    assert sample.loss_mask == trajectory.loss_mask
    assert sample.status == Sample.Status.COMPLETED and sample.reward == 1
    assert sample.response_length == len(sample.rollout_log_probs) == len(sample.loss_mask)
    assert all(p == 0 for p, m in zip(sample.rollout_log_probs, sample.loss_mask) if m == 0)


def test_async_hook_marks_sglang_abort_for_full_group_requeue(task, slime_args, monkeypatch):
    """The project hook must use Slime's ABORTED contract, not raise or reward zero."""

    class Sample:
        class Status:
            ABORTED = "aborted"

        def __init__(self, metadata):
            self.metadata = metadata
            self.status = None

    class Client:
        def __init__(self, *_args):
            pass

        async def close(self):
            pass

    async def abort_episode(*_args, **_kwargs):
        raise SGLangAbort(
            {
                "turn": 3,
                "episode": {"in_flight_episodes_at_abort": 16},
            }
        )

    slime = ModuleType("slime")
    rollout = ModuleType("slime.rollout")
    sglang_rollout = ModuleType("slime.rollout.sglang_rollout")
    sglang_rollout.GenerateState = lambda _args: SimpleNamespace(tokenizer=ByteTokenizer())
    slime.rollout, rollout.sglang_rollout = rollout, sglang_rollout
    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.rollout", rollout)
    monkeypatch.setitem(sys.modules, "slime.rollout.sglang_rollout", sglang_rollout)
    monkeypatch.setattr(slime_hooks, "SGLangClient", Client)
    monkeypatch.setattr(slime_hooks, "run_episode", abort_episode)
    events = []
    monkeypatch.setattr(slime_hooks, "report_metrics", lambda metrics: events.append(metrics))

    args = SimpleNamespace(
        **vars(slime_args), sglang_router_ip="127.0.0.1", sglang_router_port=30000
    )
    config = ExperimentConfig.from_dict(args.noise_rl)
    plan = plan_sample(config, task["id"], 0, 0)
    sample = Sample({"task": task, "noise_plan": plan.to_dict()})

    result = asyncio.run(slime_hooks.generate(args, sample, {"max_new_tokens": 96}))
    assert result is sample and sample.status == Sample.Status.ABORTED
    assert not hasattr(sample, "reward")
    assert events == [
        {
            "rollout/sglang_abort/count": 1,
            "rollout/sglang_abort/group_id": 0,
            "rollout/sglang_abort/turn": 3,
            "rollout/sglang_abort/in_flight_episodes": 16,
        }
    ]


def test_actual_sample_reward_hook(Sample, slime_args):
    cfg = ExperimentConfig()
    samples = [
        Sample(reward=float(i % 2), metadata={"noise_plan": plan_sample(cfg, "t", 0, i).to_dict()})
        for i in range(8)
    ]
    raw, values = reward_postprocess(slime_args, list(reversed(samples)))
    assert raw == [1, 0] * 4
    assert values == pytest.approx([2 / 3, -2 / 3] * 4)


def test_data_source_wraps_many_epochs_and_resumes(Sample, slime_args, tmp_path):
    path = tmp_path / "train.jsonl"
    write_records(path, mini_records(2, "train"))
    slime_args.prompt_data = str(path)
    source = NoiseDataSource(slime_args)
    groups = source.get_samples(7)
    assert len(groups) == 7 and all(len(g) == 8 for g in groups)
    assert len({s.index for g in groups for s in g}) == 56
    source.save(3)
    expected = source.get_samples(2)
    slime_args.load = slime_args.save
    resumed = NoiseDataSource(slime_args)
    resumed.load(3)
    actual = resumed.get_samples(2)
    assert [[s.metadata for s in g] for g in actual] == [[s.metadata for s in g] for g in expected]


def test_resume_config_change_rejected(Sample, slime_args, tmp_path):
    path = tmp_path / "train.jsonl"
    write_records(path, mini_records(2, "train"))
    slime_args.prompt_data = str(path)
    source = NoiseDataSource(slime_args)
    source.save(0)
    slime_args.load = slime_args.save
    slime_args.noise_rl = replace(ExperimentConfig(), seed=99).to_dict()
    with pytest.raises(ValueError, match="differs"):
        NoiseDataSource(slime_args).load(0)


@pytest.mark.parametrize(
    "option,value",
    [
        ("n_samples_per_prompt", 4),
        ("partial_rollout", True),
        ("normalize_advantages", True),
        ("over_sampling_batch_size", 3),
        ("dynamic_sampling_filter_path", "bad.filter"),
    ],
)
def test_unsafe_slime_flags_rejected(slime_args, option, value):
    setattr(slime_args, option, value)
    with pytest.raises(ValueError):
        validate_slime_args(slime_args)


def test_slime_checkout_and_exact_model_rope():
    path = os.environ.get("SLIME_SOURCE_PATH")
    if not path:
        pytest.skip("Set SLIME_SOURCE_PATH to a clean Slime checkout")
    verify_slime(path)
    arguments = model_arguments(path)
    assert arguments[arguments.index("--rotary-base") + 1] == "5000000"
    assert arguments[arguments.index("--num-layers") + 1] == "36"


def test_real_qwen_tokenizer_append_only_protocol():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            os.environ.get("QWEN_TOKENIZER", "Qwen/Qwen3-4B-Instruct-2507"), local_files_only=True
        )
    except OSError:
        pytest.skip("Download Qwen tokenizer first; this test never triggers a network download")
    protocol = QwenChatProtocol(tokenizer)
    prompt, ids = protocol.initial("Task: put an apple on the table.")
    answer = '{"action":"look"}'
    generated = tokenizer.encode(answer + "<|im_end|>", add_special_tokens=False)
    bridge = protocol.observation_segment("You see an apple.")
    joined = ids + generated + bridge.tokens
    assert tokenizer.decode(joined, skip_special_tokens=False) == prompt + answer + "<|im_end|>" + bridge.text
    assert joined[len(ids) : len(ids) + len(generated)] == generated


def test_real_alfworld_adapter_reset_step_and_isolation():
    game = os.environ.get("ALFWORLD_TEST_GAME")
    if not game:
        pytest.skip("Set ALFWORLD_TEST_GAME to a downloaded, solvable game.tw-pddl")
    pytest.importorskip("alfworld")
    from noise_rl.envs import ALFWorldEnvironment

    task = {"id": "smoke/game", "environment": "alfworld", "gamefile": str(Path(game).resolve())}
    first, second = ALFWorldEnvironment(task), ALFWorldEnvironment(task)
    try:
        initial = first.reset()
        assert initial.observation and not initial.success
        commands = first.env.state["admissible_commands"]
        move = next(c for c in commands if c.startswith("go to "))
        first.step(move)
        before = first.step("look")
        assert before.observation
        second.reset()
        second.step("look")
        after = first.step("look")
        assert before.observation == after.observation
        assert first.reset().observation == initial.observation
    finally:
        first.close()
        second.close()


def test_real_alfworld_terminal_verifier_with_test_only_planner():
    game = os.environ.get("ALFWORLD_TEST_GAME")
    if not game:
        pytest.skip("Set ALFWORLD_TEST_GAME")
    pytest.importorskip("alfworld")
    from noise_rl.envs import ALFWorldEnvironment

    env = ALFWorldEnvironment({"gamefile": game})
    try:
        # Privileged oracle is enabled ONLY in this environment-verifier test.
        # It is never used by the training adapter or rendered to the policy.
        env.env.request_infos.policy_commands = True
        state = env.env.reset()
        commands = state["policy_commands"]
        assert commands and len(commands) < 100
        for command in commands:
            result = env.step(command)
        assert result.success and result.terminated
    finally:
        env.close()
