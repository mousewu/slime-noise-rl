from collections import defaultdict

from .config import NoiseConfig
from .envs import StepResult, TextEnvironment, canonical_action
from .sampling import stable_seed

NOT_EXECUTED = "TEMPORARY_UNAVAILABLE: the operation was not executed. You may retry."
OBSERVATION_LOST = "OBSERVATION_UNAVAILABLE: the result could not be read. Check the current state."


class EventNoise:
    """A random field indexed by canonical operation and that operation's local attempt.

    Unrelated calls do not shift another operation's random stream. Marginals are
    the same for independent and coupled sampling. This is one explicit coupling,
    not a claim that all natural-language aliases have been semantically matched.
    """

    def __init__(self, seed: int, config: NoiseConfig):
        self.seed, self.config = seed, config
        self.attempts = defaultdict(int)

    def uniform(self, channel: str, action: str, attempt: int) -> float:
        return (stable_seed(self.seed, channel, action, attempt) >> 11) / 2**53

    def event(self, action: str, read_only: bool, *, canonicalized=False) -> dict:
        if not canonicalized:
            action = canonical_action(action)
        attempt = self.attempts[action]
        self.attempts[action] += 1
        block = attempt // self.config.burst_length
        burst = self.uniform("burst", action, block) < self.config.burst_probability
        drop = not read_only and (burst or self.uniform("drop", action, attempt) < self.config.action_drop)
        lost = self.uniform("observation", action, attempt) < self.config.observation_loss
        return {
            "action": action,
            "attempt": attempt,
            "dropped": drop,
            "observation_loss_draw": lost,
            "observation_lost": bool(lost and not drop),
            "burst": bool(burst and not read_only),
        }


class NoisyEnvironment:
    def __init__(self, env: TextEnvironment, config: NoiseConfig, seed: int, max_calls: int):
        self.env, self.config, self.seed, self.max_calls = env, config, seed, max_calls

    def reset(self):
        self.oracle = EventNoise(self.seed, self.config)
        self.calls = 0
        self.success = False
        self.done = False
        self.audit = []
        result = self.env.reset()
        self.success = result.success
        self.done = result.terminated or result.truncated or result.success
        return result

    def step(self, action):
        if self.done:
            raise RuntimeError("Cannot step a finished environment")
        self.calls += 1
        action = getattr(self.env, "canonical_action", canonical_action)(action)
        event = self.oracle.event(action, self.env.is_read_only(action), canonicalized=True)
        if event["dropped"]:
            result = StepResult(NOT_EXECUTED, info={"retryable": True})
        else:
            # Real infrastructure/engine errors are not synthetic failures and must propagate.
            result = self.env.step(action)
            self.success = result.success
            if event["observation_lost"]:
                result.observation = OBSERVATION_LOST
        # Terminal success remains observable through episode termination (documented semantics).
        # Do not lose the verifier's true terminal reward when the observation is hidden.
        result.success = self.success
        result.terminated = result.terminated or self.success
        result.truncated = result.truncated or (self.calls >= self.max_calls and not result.terminated)
        self.done = result.terminated or result.truncated
        event.update(success=self.success, call=self.calls)
        self.audit.append(event)
        return result

    def close(self):
        self.env.close()


def execute_with_retry(env: NoisyEnvironment, action: str, retry_limit: int) -> StepResult:
    """Conservative baseline: retry only publicly reported definitely-not-executed calls.

    Hidden fault annotations are NOT consulted. Every retry counts toward the tool budget.
    Ambiguous outcomes are left to the agent; blindly retrying them can duplicate effects.
    """
    observations = []
    for attempt in range(retry_limit + 1):
        result = env.step(action)
        observations.append(result.observation)
        if result.terminated or result.truncated or result.observation != NOT_EXECUTED:
            break
    result.observation = "\n".join(observations)
    return result
