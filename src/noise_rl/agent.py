import math
import time
from dataclasses import dataclass, field

import httpx

from .config import ExperimentConfig
from .envs import make_environment, parse_action
from .noise import NoisyEnvironment, execute_with_retry
from .sampling import SamplingPlan, stable_seed

SYSTEM_PROMPT = """You control a text household environment. Complete the task using its commands.
Reply with exactly one JSON object: {"action":"<one environment command>"}.
Do not emit explanations, multiple actions, or invented observations.
Useful commands include look, inventory, go to <receptacle>, open <receptacle>,
take <object> from <receptacle>, put <object> in/on <receptacle>,
clean <object> with <receptacle>, heat <object> with <receptacle>,
cool <object> with <receptacle>, and use <object>.
Tools can fail. If an outcome is uncertain, inspect the current state before deciding what to do.
The episode ends when the environment verifies the goal or the interaction budget is exhausted."""


@dataclass
class Generation:
    tokens: list[int]
    log_probs: list[float]
    text: str
    finish_reason: str
    meta_info: dict = field(default_factory=dict)


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

    @property
    def tokens(self):
        return self.prompt_tokens + [t for segment in self.segments for t in segment.tokens]

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
        for text, token in (("<|im_end|>", self.end_id), ("<|im_start|>", self.start_id)):
            if not isinstance(token, int) or tokenizer.encode(text, add_special_tokens=False) != [token]:
                raise ValueError("This implementation requires a Qwen im_start/im_end tokenizer")

    @staticmethod
    def safe_observation(text):
        # Environment text must not be interpreted as chat control tokens.
        return text.replace("<|", "< |")

    def initial(self, observation):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.safe_observation(observation)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        # The selected Instruct-2507 template must end directly at the assistant prefix.
        if not text.endswith("<|im_start|>assistant\n"):
            raise ValueError("Unsupported chat template: use Qwen3-4B-Instruct-2507, not a Thinking template")
        return text, self.tokenizer.encode(text, add_special_tokens=False)

    def observation_segment(self, observation):
        text = (
            "\n<|im_start|>user\n"
            + self.safe_observation(observation)
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        return Segment(self.tokenizer.encode(text, add_special_tokens=False), None, text, False)

    def action_text(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


class SGLangClient:
    def __init__(self, url: str, timeout: float = 180, headers=None):
        self.url = url.rstrip("/") + "/generate"
        self.client = httpx.AsyncClient(timeout=timeout, headers=headers, trust_env=False)

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
        pairs = meta.get("output_token_logprobs")
        if not pairs:
            raise RuntimeError("SGLang returned no output token logprobs; cannot train on re-tokenized text")
        ids, probs = [p[1] for p in pairs], [p[0] for p in pairs]
        if not all(type(t) is int and t >= 0 for t in ids):
            raise ValueError("Invalid output token ids")
        if not all(isinstance(p, (int, float)) and math.isfinite(p) and p <= 1e-5 for p in probs):
            raise ValueError("Invalid rollout log probabilities")
        finish = meta.get("finish_reason", {})
        reason = finish.get("type") if isinstance(finish, dict) else finish
        return Generation(ids, probs, data.get("text", ""), reason, meta)

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
    protocol = QwenChatProtocol(tokenizer)
    env = NoisyEnvironment(
        environment_factory(task), config.noise, plan.environment_seed, config.max_tool_calls
    )
    try:
        initial = env.reset()
        if initial.success or initial.terminated or initial.truncated:
            raise ValueError("Task must start in a nonterminal, unsolved state")
        prompt, prompt_tokens = protocol.initial(initial.observation)
        if len(prompt_tokens) >= config.max_context_tokens:
            raise ValueError(f"Initial prompt exceeds the context budget: {task['id']}")
        trajectory = Trajectory(prompt, prompt_tokens)
        params = dict(sampling_params or {"temperature": 0.8, "top_p": 1.0, "top_k": -1})
        if params.get("top_p", 1.0) != 1.0 or params.get("top_k", -1) != -1:
            raise ValueError("First version requires top_p=1, top_k=-1 for comparable full-support rollouts")
        # Engine sampling limits and protocol stops cannot be overridden by dataset params.
        params.pop("stop", None)
        params.pop("min_new_tokens", None)
        params.update(stop_token_ids=[protocol.end_id], no_stop_trim=True, skip_special_tokens=False)
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
            generated = await client.generate(token_ids, call_params)
            if not generated.tokens or len(generated.tokens) != len(generated.log_probs):
                raise ValueError("Unaligned/empty generation")
            if len(generated.tokens) > remaining:
                raise ValueError("Inference server exceeded the requested token budget")
            trajectory.inference_input_tokens += len(token_ids)
            trajectory.segments.append(
                Segment(generated.tokens, generated.log_probs, generated.text, True, generated.meta_info)
            )
            if generated.finish_reason == "abort":
                raise RuntimeError(
                    "Inference was aborted; do not turn infrastructure failures into zero reward"
                )
            if generated.finish_reason == "length":
                trajectory.termination = "generation_length"
                break  # Never execute a truncated tool call.
            if generated.finish_reason != "stop" or generated.tokens[-1] != protocol.end_id:
                raise ValueError("Expected retained im_end stop token in SGLang token/logprob response")
            response_text = protocol.action_text(generated.tokens)
            try:
                action = parse_action(response_text)
            except ValueError as exc:
                observation = f"FORMAT_ERROR: {exc}"
                trajectory.steps.append(
                    {"turn": turn, "action": None, "format_error": True, "observation": observation}
                )
            else:
                result = execute_with_retry(env, action, config.retry_limit)
                observation = result.observation
                trajectory.steps.append(
                    {"turn": turn, "action": action, "format_error": False, "observation": observation}
                )
                trajectory.success = result.success
                if result.terminated or result.truncated:
                    trajectory.termination = (
                        "success"
                        if result.success
                        else ("tool_budget" if result.truncated else "environment_terminal")
                    )
                    break
            if turn + 1 == config.max_turns:
                trajectory.termination = "turn_budget"
                break
            bridge = protocol.observation_segment(observation)
            if len(trajectory.tokens) + len(bridge.tokens) >= config.max_context_tokens:
                trajectory.termination = "context_budget"
                break  # No silent history truncation or retokenization.
            trajectory.segments.append(bridge)
        trajectory.audit = list(env.audit)
        trajectory.tool_calls = env.calls
        trajectory.elapsed_seconds = time.monotonic() - started
        return trajectory
    finally:
        env.close()
