import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class StepResult:
    observation: str
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    info: dict = field(default_factory=dict)  # Trainer-only; NEVER rendered into model context.


class TextEnvironment(Protocol):
    def reset(self) -> StepResult: ...
    def step(self, action: str) -> StepResult: ...
    def is_read_only(self, action: str) -> bool: ...
    def close(self) -> None: ...


def canonical_action(action: str) -> str:
    return " ".join(action.strip().lower().split())


def parse_action(text: str) -> str:
    """Strict single-action JSON; no regex salvage of hallucinated extra turns."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError('Return exactly one JSON object: {"action":"<command>"}') from exc
    if not isinstance(value, dict) or set(value) != {"action"}:
        raise ValueError('Expected exactly one key: "action"')
    if not isinstance(value["action"], str) or not value["action"].strip():
        raise ValueError("action must be a nonempty string")
    if len(value["action"]) > 512 or "\n" in value["action"] or "\r" in value["action"]:
        raise ValueError("Action must be one short command")
    return canonical_action(value["action"])


class MiniHousehold:
    """Dependency-free plumbing fixture, NOT ALFWorld or a publication benchmark."""

    def __init__(self, task: dict):
        self.task = task
        self.item = task.get("item", "apple 1")
        self.target = task.get("target", "table 1")
        if not all(re.fullmatch(r"[a-z]+ [1-9][0-9]*", x) for x in (self.item, self.target)):
            raise ValueError("Invalid fixture object names")

    def reset(self):
        self.location = "shelf 1"
        self.clean = False
        return StepResult(
            f"Task: clean {self.item} and put it on {self.target}. "
            f"You see {self.item} on shelf 1. A sinkbasin 1 and {self.target} are accessible. "
            "Commands: look; inventory; take <object> from <place>; "
            "clean <object> with sinkbasin 1; put <object> in/on <place>."
        )

    def is_read_only(self, action):
        return action in {"look", "inventory"}

    def step(self, action):
        action = canonical_action(action)
        if action == "look":
            feedback = f"{self.item} is at {self.location}; clean={str(self.clean).lower()}."
        elif action == "inventory":
            feedback = f"Holding: {self.item if self.location == 'inventory' else 'nothing'}."
        elif action == f"take {self.item} from {self.location}" and self.location != "inventory":
            self.location = "inventory"
            feedback = f"You take {self.item}."
        elif action == f"clean {self.item} with sinkbasin 1" and self.location == "inventory":
            self.clean = True
            feedback = f"You clean {self.item}."
        elif action == f"put {self.item} in/on {self.target}" and self.location == "inventory":
            self.location = self.target
            feedback = f"You put {self.item} on {self.target}."
        else:
            feedback = "Nothing happens. Invalid action in the current state."
        success = self.location == self.target and self.clean
        return StepResult(feedback, success=success, terminated=success)

    def close(self):
        pass


class ALFWorldEnvironment:
    """One explicitly selected game, no expert plans, no shuffled object identifiers."""

    def __init__(self, task: dict):
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        game = Path(task["gamefile"]).expanduser().resolve(strict=True)
        if game.suffix != ".tw-pddl":
            raise ValueError("Expected an ALFWorld game.tw-pddl file")
        if task.get("game_sha256"):
            import hashlib

            if hashlib.sha256(game.read_bytes()).hexdigest() != task["game_sha256"]:
                raise ValueError(f"Game content changed since manifest creation: {game}")
        # Direct TextWorld API avoids registering a fresh global Gym id for every trajectory.
        self.env = textworld.start(
            str(game),
            request_infos=textworld.EnvInfos(won=True),
            wrappers=[AlfredDemangler(shuffle=False)],
        )

    def reset(self):
        state = self.env.reset()
        return StepResult(str(state.feedback), success=bool(state.get("won", False)))

    def is_read_only(self, action):
        return action in {"look", "inventory"} or action.startswith("examine ")

    def step(self, action):
        state, _score, done = self.env.step(action)
        return StepResult(str(state.feedback), success=bool(state.get("won", False)), terminated=bool(done))

    def close(self):
        self.env.close()


def make_environment(task: dict) -> TextEnvironment:
    backend = task["environment"]
    if backend == "mini":
        return MiniHousehold(task)
    if backend == "alfworld":
        return ALFWorldEnvironment(task)
    raise ValueError(f"Unsupported environment: {backend}")
