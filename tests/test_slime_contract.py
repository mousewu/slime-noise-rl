"""Runs against actual upstream Sample implementation, not a replacement dataclass."""

import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest

from noise_rl.agent import QwenChatProtocol, run_episode
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


def test_pinned_slime_and_exact_model_rope():
    path = os.environ.get("SLIME_SOURCE_PATH")
    if not path:
        pytest.skip("Set SLIME_SOURCE_PATH to a clean pinned Slime checkout")
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
