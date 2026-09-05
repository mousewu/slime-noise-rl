import pytest

from noise_rl.config import NoiseConfig
from noise_rl.envs import MiniHousehold, parse_action
from noise_rl.noise import NOT_EXECUTED, OBSERVATION_LOST, EventNoise, NoisyEnvironment, execute_with_retry


def test_clean_fixture_success(task):
    env = NoisyEnvironment(MiniHousehold(task), NoiseConfig(0, 0), 42, 20)
    env.reset()
    for action in [
        "take apple 1 from shelf 1",
        "clean apple 1 with sinkbasin 1",
        "put apple 1 in/on table 1",
    ]:
        result = env.step(action)
    assert result.success and result.terminated
    with pytest.raises(RuntimeError):
        env.step("look")


def test_unrelated_calls_do_not_shift_failure_stream():
    a, b = EventNoise(1234, NoiseConfig()), EventNoise(1234, NoiseConfig())
    for _ in range(12):
        expected = a.event("take apple 1 from shelf 1", False)
        b.event("look", True)
        b.event("inventory", True)
        assert expected == b.event("  TAKE   APPLE 1 from shelf 1 ", False)


def test_observation_loss_does_not_erase_terminal_reward(task):
    env = NoisyEnvironment(MiniHousehold(task), NoiseConfig(0, 1), 0, 10)
    env.reset()
    for action in [
        "take apple 1 from shelf 1",
        "clean apple 1 with sinkbasin 1",
        "put apple 1 in/on table 1",
    ]:
        result = env.step(action)
        assert result.observation == OBSERVATION_LOST
    assert result.success and result.terminated


def test_harness_retry_is_budgeted_and_uses_public_output(task):
    env = NoisyEnvironment(MiniHousehold(task), NoiseConfig(1, 0), 0, 2)
    env.reset()
    result = execute_with_retry(env, "take apple 1 from shelf 1", 10)
    assert env.calls == 2 and result.truncated and not result.success
    assert result.observation.count(NOT_EXECUTED) == 2


def test_read_only_queries_not_dropped(task):
    env = NoisyEnvironment(MiniHousehold(task), NoiseConfig(1, 0), 0, 10)
    env.reset()
    assert "is at shelf 1" in env.step("look").observation
    assert "nothing" in env.step("inventory").observation


def test_burst_block(task):
    oracle = EventNoise(0, NoiseConfig(0, 0, burst_probability=1, burst_length=3))
    assert all(oracle.event("take apple", False)["dropped"] for _ in range(12))


def test_fault_rates_match_declared_marginals():
    events = [EventNoise(i, NoiseConfig(0.2, 0.3)).event("take apple", False) for i in range(5000)]
    assert sum(e["dropped"] for e in events) / len(events) == pytest.approx(0.2, abs=0.02)
    assert sum(e["observation_loss_draw"] for e in events) / len(events) == pytest.approx(0.3, abs=0.02)
    executed = [e for e in events if not e["dropped"]]
    assert sum(e["observation_lost"] for e in executed) / len(executed) == pytest.approx(0.3, abs=0.02)


def test_reset_replays_same_environment(task):
    env = NoisyEnvironment(MiniHousehold(task), NoiseConfig(), 333, 10)
    sequences = []
    for _ in range(2):
        env.reset()
        sequences.append(
            [env.step(a).observation for a in ["take apple 1 from shelf 1", "look", "inventory"]]
        )
    assert sequences[0] == sequences[1]


@pytest.mark.parametrize(
    "text",
    [
        '{"action":"look"} junk',
        '{"action":"look","thought":"x"}',
        '[{"action":"look"}]',
        '{"action":1}',
        '{"action":""}',
        '{"action":"look\\ninventory"}',
        '```json\n{"action":"look"}\n```',
    ],
)
def test_strict_action_parser(text):
    with pytest.raises(ValueError):
        parse_action(text)
