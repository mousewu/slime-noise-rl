import asyncio
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock

import httpx

from .config import ExperimentConfig
from .environment_runners import (
    ProcessIsolatedALFWorldEnvironment,
    get_process_environment_pool,
)
from .envs import ALFWorldEnvironment, make_environment, parse_action
from .noise import NoisyEnvironment, execute_with_retry
from .rollout_diagnostics import weight_update_snapshot
from .sampling import SamplingPlan, stable_seed


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You control a text household environment. Complete the task using its commands.
Reply with exactly one JSON object: {"action":"<one environment command>"}.
Do not emit explanations, multiple actions, or invented observations.
Useful commands include look, inventory, go to <receptacle>, open <receptacle>,
take <object> from <receptacle>, put <object> in/on <receptacle>,
clean <object> with <receptacle>, heat <object> with <receptacle>,
cool <object> with <receptacle>, and use <object>.
Tools can fail. If an outcome is uncertain, inspect the current state before deciding what to do.
The episode ends when the environment verifies the goal or the interaction budget is exhausted."""


_ENVIRONMENT_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_ENVIRONMENT_EXECUTORS_LOCK = Lock()
_EPISODE_ACTIVITY_LOCK = Lock()
_TEXTWORLD_STEP_LOCK = Lock()
_IN_FLIGHT_EPISODES = 0


def in_flight_episode_count() -> int:
    """Return the current process-local number of live agent trajectories."""
    with _EPISODE_ACTIVITY_LOCK:
        return _IN_FLIGHT_EPISODES


def environment_executor(workers: int) -> ThreadPoolExecutor:
    """Return the process-local, bounded executor for independent environment steps."""
    with _ENVIRONMENT_EXECUTORS_LOCK:
        executor = _ENVIRONMENT_EXECUTORS.get(workers)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="noise-rl-env"
            )
            _ENVIRONMENT_EXECUTORS[workers] = executor
        return executor


def _is_alfworld_environment(env: NoisyEnvironment) -> bool:
    """Whether a (possibly noisy) environment uses TextWorld underneath."""
    return isinstance(getattr(env, "env", env), ALFWorldEnvironment)


def _environment_dispatch_workers(env: NoisyEnvironment, workers: int) -> int:
    """Keep enough RPC threads to occupy every leased process runner."""
    raw_environment = getattr(env, "env", env)
    if isinstance(raw_environment, ProcessIsolatedALFWorldEnvironment):
        return max(workers, raw_environment.runner_count)
    return workers


def create_environment(task: dict, environment_factory):
    """Create a TextWorld game while holding its process-global parser lock."""
    if task.get("environment") == "alfworld":
        with _TEXTWORLD_STEP_LOCK:
            return environment_factory(task)
    return environment_factory(task)


def reset_environment(env: NoisyEnvironment):
    """Reset a TextWorld game while no worker thread is parsing another game."""
    if _is_alfworld_environment(env):
        with _TEXTWORLD_STEP_LOCK:
            return env.reset()
    return env.reset()


def close_environment(env: NoisyEnvironment) -> None:
    """Close a TextWorld game while no worker thread is using its parser state."""
    if _is_alfworld_environment(env):
        with _TEXTWORLD_STEP_LOCK:
            env.close()
    else:
        env.close()


class EpisodeActivity:
    """Track local concurrent trajectories without changing their per-episode order."""

    def __enter__(self) -> int:
        global _IN_FLIGHT_EPISODES
        with _EPISODE_ACTIVITY_LOCK:
            _IN_FLIGHT_EPISODES += 1
            return _IN_FLIGHT_EPISODES

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _IN_FLIGHT_EPISODES
        with _EPISODE_ACTIVITY_LOCK:
            _IN_FLIGHT_EPISODES -= 1
            if _IN_FLIGHT_EPISODES < 0:  # Defensive: never conceal a lifecycle bug.
                _IN_FLIGHT_EPISODES = 0
                raise RuntimeError("Episode activity counter became negative")


def _execute_environment_step(env: NoisyEnvironment, action: str, retry_limit: int):
    # TextWorld's Tatsu grammar parser is process-global and mutable.  Separate
    # ALFWorld instances cannot call it concurrently from Python threads.
    # Keep the event loop unblocked, but serialize only this unsafe backend.
    step_lock = _TEXTWORLD_STEP_LOCK if _is_alfworld_environment(env) else None
    if step_lock is None:
        started = time.monotonic()
        result = execute_with_retry(env, action, retry_limit)
    else:
        with step_lock:
            started = time.monotonic()
            result = execute_with_retry(env, action, retry_limit)
    return result, started, time.monotonic()


async def execute_environment_step(
    env: NoisyEnvironment, action: str, retry_limit: int, workers: int
):
    """Run one independent environment interaction without blocking the rollout loop.

    A trajectory awaits its own result before its next action, so environment
    state transitions remain strictly sequential.  Only distinct trajectories
    can occupy different executor threads.
    """
    queued = time.monotonic()
    loop = asyncio.get_running_loop()
    result, started, finished = await loop.run_in_executor(
        environment_executor(_environment_dispatch_workers(env, workers)),
        _execute_environment_step,
        env,
        action,
        retry_limit,
    )
    return result, started - queued, finished - started


async def create_episode_environment(
    task: dict,
    config: ExperimentConfig,
    environment_factory,
):
    """Create a backend without blocking the rollout loop.

    Process runners are deliberately limited to real ALFWorld tasks produced by
    the built-in factory.  Test fixtures and other backends keep the established
    in-process semantics.
    """
    if task.get("environment") == "awm":
        from .awm import AWMEnvironment

        if not config.awm_url:
            raise ValueError("AWM tasks require noise_rl.awm_url")
        return AWMEnvironment(task, config.awm_url, config.request_timeout), 0.0
    if (
        config.environment_processes
        and task.get("environment") == "alfworld"
        and environment_factory is make_environment
    ):
        return await get_process_environment_pool(config.environment_processes).open(task)
    loop = asyncio.get_running_loop()
    environment = await loop.run_in_executor(
        environment_executor(config.environment_workers),
        create_environment,
        task,
        environment_factory,
    )
    return environment, 0.0


async def reset_episode_environment(env: NoisyEnvironment, workers: int):
    """Reset through bounded workers so a runner RPC cannot stall asyncio."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        environment_executor(_environment_dispatch_workers(env, workers)),
        reset_environment,
        env,
    )


async def close_episode_environment(env: NoisyEnvironment, workers: int) -> None:
    """Return a process runner lease without blocking unrelated trajectories."""
    raw_environment = getattr(env, "env", env)
    if isinstance(raw_environment, ProcessIsolatedALFWorldEnvironment):
        await raw_environment.aclose()
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        environment_executor(_environment_dispatch_workers(env, workers)),
        close_environment,
        env,
    )


@dataclass
class Generation:
    tokens: list[int]
    log_probs: list[float]
    text: str
    finish_reason: str
    meta_info: dict = field(default_factory=dict)
    request_id: str | None = None


class SGLangAbort(RuntimeError):
    """A weight-sync interruption that must be retried as a complete group.

    This is deliberately distinct from transport and server failures.  Slime's
    fully-async worker recognizes ``Sample.Status.ABORTED`` and puts the whole
    comparison group back in its data buffer.  The custom generation hook
    converts this exception into that status; it must never become reward zero.
    """

    def __init__(self, diagnostics: dict):
        super().__init__("SGLang inference aborted; requeue the full comparison group")
        self.diagnostics = diagnostics


@dataclass
class Segment:
    tokens: list[int]
    log_probs: list[float] | None
    text: str
    trainable: bool
    meta_info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    prompt_text: str
    prompt_tokens: list[int]
    segments: list[Segment] = field(default_factory=list)
    success: bool = False
    termination: str = "pending"
    steps: list[dict] = field(default_factory=list)
    audit: list[dict] = field(default_factory=list)
    inference_input_tokens: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0
    model_requests: int = 0
    model_request_seconds: float = 0.0
    environment_queue_seconds: float = 0.0
    environment_step_seconds: float = 0.0
    environment_runner_wait_seconds: float = 0.0
    in_flight_episodes_at_start: int = 0

    @property
    def tokens(self):
        return self.prompt_tokens + [
            t for segment in self.segments for t in segment.tokens
        ]

    @property
    def generated_tokens(self):
        return sum(len(s.tokens) for s in self.segments if s.trainable)

    @property
    def loss_mask(self):
        return [int(s.trainable) for s in self.segments for _ in s.tokens]


class QwenChatProtocol:
    """Append-only Qwen3 chat framing: never re-tokenize previously generated text.

    Explicitly limited to Qwen-style im_start/im_end chat templates. Other model
    families need their own protocol and prefix-equivalence tests.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        for text, token in (
            ("<|im_end|>", self.end_id),
            ("<|im_start|>", self.start_id),
        ):
            if not isinstance(token, int) or tokenizer.encode(
                text, add_special_tokens=False
            ) != [token]:
                raise ValueError(
                    "This implementation requires a Qwen im_start/im_end tokenizer"
                )

    @staticmethod
    def safe_observation(text):
        # Environment text must not be interpreted as chat control tokens.
        return text.replace("<|", "< |")

    def initial(self, observation, system_prompt=SYSTEM_PROMPT):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.safe_observation(observation)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        # The selected Instruct-2507 template must end directly at the assistant prefix.
        if not text.endswith("<|im_start|>assistant\n"):
            raise ValueError(
                "Unsupported chat template: use Qwen3-4B-Instruct-2507, not a Thinking template"
            )
        return text, self.tokenizer.encode(text, add_special_tokens=False)

    def observation_segment(self, observation):
        text = (
            "\n<|im_start|>user\n"
            + self.safe_observation(observation)
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        return Segment(
            self.tokenizer.encode(text, add_special_tokens=False), None, text, False
        )

    def action_text(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


class SGLangClient:
    def __init__(self, url: str, timeout: float = 180, headers=None):
        self.url = url.rstrip("/") + "/generate"
        self.client = httpx.AsyncClient(
            timeout=timeout, headers=headers, trust_env=False
        )

    async def generate(self, tokens, sampling_params):
        response = await self.client.post(
            self.url,
            json={
                "input_ids": tokens,
                "sampling_params": sampling_params,
                "return_logprob": True,
                "logprob_start_len": -1,
            },
        )
        response.raise_for_status()
        data = response.json()
        meta = data["meta_info"]
        finish = meta.get("finish_reason", {})
        reason = finish.get("type") if isinstance(finish, dict) else finish
        request_id = (
            data.get("id")
            or meta.get("request_id")
            or meta.get("id")
            or meta.get("rid")
            or response.headers.get("x-request-id")
        )

        # SGLang can abort before it has generated any token/logprob when the
        # actor publishes a new weight version.  Return this separately from a
        # malformed successful response so run_episode can preserve the
        # diagnostics and the fully-async hook can requeue the *whole* group.
        # Partial tokens are never resumed or trained after a policy change.
        if reason == "abort":
            return Generation(
                [],
                [],
                data.get("text", ""),
                reason,
                meta,
                str(request_id) if request_id is not None else None,
            )

        pairs = meta.get("output_token_logprobs")
        if not pairs:
            raise RuntimeError(
                "SGLang returned no output token logprobs; cannot train on re-tokenized text"
            )
        ids, probs = [p[1] for p in pairs], [p[0] for p in pairs]
        if not all(type(t) is int and t >= 0 for t in ids):
            raise ValueError("Invalid output token ids")
        if not all(
            isinstance(p, (int, float)) and math.isfinite(p) and p <= 1e-5
            for p in probs
        ):
            raise ValueError("Invalid rollout log probabilities")
        return Generation(
            ids,
            probs,
            data.get("text", ""),
            reason,
            meta,
            str(request_id) if request_id is not None else None,
        )

    async def close(self):
        await self.client.aclose()


async def run_episode(
    task: dict,
    plan: SamplingPlan,
    config: ExperimentConfig,
    tokenizer,
    client,
    sampling_params: dict | None = None,
    environment_factory=make_environment,
) -> Trajectory:
    started = time.monotonic()
    with EpisodeActivity() as in_flight_at_start:
        protocol = QwenChatProtocol(tokenizer)
        env = None
        try:
            environment, runner_wait_seconds = await create_episode_environment(
                task, config, environment_factory
            )
            env = NoisyEnvironment(
                environment,
                config.noise,
                plan.environment_seed,
                config.max_tool_calls,
            )
            initial = await reset_episode_environment(env, config.environment_workers)
            if initial.success or initial.terminated or initial.truncated:
                raise ValueError("Task must start in a nonterminal, unsolved state")
            if task.get("environment") == "awm":
                from .awm import SYSTEM_PROMPT as awm_prompt
                prompt, prompt_tokens = protocol.initial(initial.observation, awm_prompt)
            else:
                prompt, prompt_tokens = protocol.initial(initial.observation)
            if len(prompt_tokens) >= config.max_context_tokens:
                raise ValueError(
                    f"Initial prompt exceeds the context budget: {task['id']}"
                )
            trajectory = Trajectory(
                prompt,
                prompt_tokens,
                in_flight_episodes_at_start=in_flight_at_start,
                environment_runner_wait_seconds=runner_wait_seconds,
            )
            params = dict(
                sampling_params or {"temperature": 0.8, "top_p": 1.0, "top_k": -1}
            )
            if params.get("top_p", 1.0) != 1.0 or params.get("top_k", -1) != -1:
                raise ValueError(
                    "First version requires top_p=1, top_k=-1 for comparable full-support rollouts"
                )
            # Engine sampling limits and protocol stops cannot be overridden by dataset params.
            params.pop("stop", None)
            params.pop("min_new_tokens", None)
            params.update(
                stop_token_ids=[protocol.end_id],
                no_stop_trim=True,
                skip_special_tokens=False,
            )
            for turn in range(config.max_turns):
                token_ids = trajectory.tokens
                remaining = min(
                    config.max_generated_tokens - trajectory.generated_tokens,
                    config.max_context_tokens - len(token_ids),
                    config.max_tokens_per_turn,
                )
                if remaining <= 0:
                    trajectory.termination = "token_budget"
                    break
                call_params = dict(
                    params,
                    max_new_tokens=remaining,
                    sampling_seed=stable_seed(plan.policy_seed, turn) % (2**31 - 1),
                )
                model_started = time.monotonic()
                generated = await client.generate(token_ids, call_params)
                request_elapsed_seconds = time.monotonic() - model_started
                trajectory.model_requests += 1
                trajectory.model_request_seconds += request_elapsed_seconds
                if generated.finish_reason == "abort":
                    # Do this before validating or appending generation tokens:
                    # a response may have zero or partial tokens after a weight
                    # update, neither of which belongs in an on-policy trace.
                    abort_diagnostics = {
                        "event": "sglang_inference_abort",
                        "task_id": plan.task_id,
                        "group_id": plan.group_id,
                        "rank": plan.rank,
                        "scenario_id": plan.scenario_id,
                        "evaluation": plan.evaluation,
                        "turn": turn,
                        "request": {
                            "request_id": generated.request_id,
                            "url": getattr(client, "url", None),
                            "finish_reason": generated.finish_reason,
                            "meta_info": generated.meta_info,
                            "input_tokens": len(token_ids),
                            "output_tokens": len(generated.tokens),
                            "max_new_tokens": remaining,
                            "elapsed_seconds": request_elapsed_seconds,
                            "episode_elapsed_seconds": trajectory.model_request_seconds,
                        },
                        "episode": {
                            "model_requests": trajectory.model_requests,
                            "in_flight_episodes_at_start": in_flight_at_start,
                            "in_flight_episodes_at_abort": in_flight_episode_count(),
                        },
                        "weight_update": weight_update_snapshot(),
                    }
                    logger.warning(
                        "SGLang abort diagnostics (full group will be requeued): %s",
                        json.dumps(abort_diagnostics, ensure_ascii=False, sort_keys=True, default=str),
                    )
                    raise SGLangAbort(abort_diagnostics)
                if not generated.tokens or len(generated.tokens) != len(
                    generated.log_probs
                ):
                    raise ValueError("Unaligned/empty generation")
                if len(generated.tokens) > remaining:
                    raise ValueError(
                        "Inference server exceeded the requested token budget"
                    )
                trajectory.inference_input_tokens += len(token_ids)
                trajectory.segments.append(
                    Segment(
                        generated.tokens,
                        generated.log_probs,
                        generated.text,
                        True,
                        generated.meta_info,
                    )
                )
                if generated.finish_reason == "length":
                    trajectory.termination = "generation_length"
                    break  # Never execute a truncated tool call.
                if (
                    generated.finish_reason != "stop"
                    or generated.tokens[-1] != protocol.end_id
                ):
                    raise ValueError(
                        "Expected retained im_end stop token in SGLang token/logprob response"
                    )
                response_text = protocol.action_text(generated.tokens)
                try:
                    if task.get("environment") == "awm":
                        from .awm import canonical_action
                        action = canonical_action(response_text)
                    else:
                        action = parse_action(response_text)
                except ValueError as exc:
                    observation = f"FORMAT_ERROR: {exc}"
                    trajectory.steps.append(
                        {
                            "turn": turn,
                            "action": None,
                            "format_error": True,
                            "observation": observation,
                        }
                    )
                else:
                    result, queued_seconds, environment_seconds = (
                        await execute_environment_step(
                            env, action, config.retry_limit, config.environment_workers
                        )
                    )
                    trajectory.environment_queue_seconds += queued_seconds
                    trajectory.environment_step_seconds += environment_seconds
                    observation = result.observation
                    trajectory.steps.append(
                        {
                            "turn": turn,
                            "action": action,
                            "format_error": False,
                            "observation": observation,
                        }
                    )
                    trajectory.success = result.success
                    if result.terminated or result.truncated:
                        trajectory.termination = (
                            "success"
                            if result.success
                            else (
                                "tool_budget"
                                if result.truncated
                                else "environment_terminal"
                            )
                        )
                        break
                if turn + 1 == config.max_turns:
                    trajectory.termination = "turn_budget"
                    break
                bridge = protocol.observation_segment(observation)
                if (
                    len(trajectory.tokens) + len(bridge.tokens)
                    >= config.max_context_tokens
                ):
                    trajectory.termination = "context_budget"
                    break  # No silent history truncation or retokenization.
                trajectory.segments.append(bridge)
            trajectory.audit = list(env.audit)
            trajectory.tool_calls = env.calls
            trajectory.elapsed_seconds = time.monotonic() - started
            return trajectory
        finally:
            if env is not None:
                await close_episode_environment(env, config.environment_workers)
