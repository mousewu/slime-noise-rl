"""Explicit CPU-only fixtures. Results from these are not LLM/RL experiments."""

import json
import random

from .agent import Generation
from .sampling import stable_seed


class ByteTokenizer:
    start, end = 100001, 100002

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": self.start, "<|im_end|>": self.end}.get(token)

    def encode(self, text, add_special_tokens=False):
        output = []
        while text:
            for special, token in (("<|im_start|>", self.start), ("<|im_end|>", self.end)):
                if text.startswith(special):
                    output.append(token)
                    text = text[len(special) :]
                    break
            else:
                output.append(ord(text[0]))
                text = text[1:]
        return output

    def decode(self, tokens, skip_special_tokens=False):
        special = {self.start: "<|im_start|>", self.end: "<|im_end|>"}
        return "".join(
            ("" if skip_special_tokens else special[t]) if t in special else chr(t) for t in tokens
        )

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text


class ScriptedClient:
    """Small observation-driven policy; has no access to fault seeds, audit, or verifier state."""

    def __init__(self, task, policy_seed=0, competence=1.0):
        self.tokenizer = ByteTokenizer()
        self.item, self.target = task["item"], task["target"]
        self.competent = random.Random(stable_seed(policy_seed, "fixture-choice")).random() < competence
        self.holding, self.clean, self.last = False, False, None

    async def generate(self, tokens, sampling_params):
        text = self.tokenizer.decode(tokens)
        observation = text.rsplit("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
        if "TEMPORARY_UNAVAILABLE" not in observation and "OBSERVATION_UNAVAILABLE" not in observation:
            if f"You take {self.item}" in observation:
                self.holding = True
            if f"You clean {self.item}" in observation:
                self.clean = True
            if " is at inventory;" in observation:
                self.holding = True
            if "clean=true" in observation:
                self.clean = True
        if "OBSERVATION_UNAVAILABLE" in observation:
            action = "look"
        elif not self.holding:
            action = f"take {self.item} from shelf 1"
        elif self.competent and not self.clean:
            action = f"clean {self.item} with sinkbasin 1"
        else:
            action = f"put {self.item} in/on {self.target}"
        answer = json.dumps({"action": action}, separators=(",", ":")) + "<|im_end|>"
        generated = self.tokenizer.encode(answer)
        limit = sampling_params["max_new_tokens"]
        reason = "stop" if len(generated) <= limit else "length"
        generated = generated[:limit]
        return Generation(generated, [-0.1] * len(generated), self.tokenizer.decode(generated), reason)

    async def close(self):
        pass
