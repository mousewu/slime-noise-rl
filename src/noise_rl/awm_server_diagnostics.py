"""Runtime diagnostics for OpenEnv's process-isolated AWM scenarios.

OpenEnv converts sub-environment MCP exceptions into ``server_error``
observations.  The original FastAPI traceback lives only in a per-session
``server.log`` which is normally cleaned up as soon as the WebSocket session
closes.  This module patches the running AWM server in memory, before Uvicorn
loads its app, and emits a bounded structured record while that file still
exists.  It deliberately has only standard-library dependencies so it can run
inside the isolated NumPy-2 AWM environment.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4


_LOGGER = logging.getLogger(__name__)
_DEFAULT_LOG_TAIL_BYTES = 24 * 1024
_MAX_LOG_TAIL_BYTES = 256 * 1024
_SECRET_NAMES = (
    "SWANLAB_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENENV_AWM_LLM_API_KEY",
)


def _diagnostics_directory() -> Path | None:
    configured = os.environ.get("NOISE_RL_AWM_DIAGNOSTICS_DIR")
    if not configured:
        return None
    directory = Path(configured).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _tail_bytes() -> int:
    raw = os.environ.get("NOISE_RL_AWM_SUBPROCESS_LOG_TAIL_BYTES")
    if raw is None:
        return _DEFAULT_LOG_TAIL_BYTES
    try:
        value = int(raw)
    except ValueError:
        _LOGGER.warning(
            "Ignoring invalid NOISE_RL_AWM_SUBPROCESS_LOG_TAIL_BYTES=%r; using %d",
            raw,
            _DEFAULT_LOG_TAIL_BYTES,
        )
        return _DEFAULT_LOG_TAIL_BYTES
    if value <= 0:
        return 0
    return min(value, _MAX_LOG_TAIL_BYTES)


def _redact(value: str) -> str:
    for name in _SECRET_NAMES:
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, f"<{name}_REDACTED>")
    return value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _read_subprocess_log(process: Any) -> dict[str, Any]:
    path = _field(process, "_log_path")
    log_file = _field(process, "_log_file")
    if log_file is not None:
        try:
            log_file.flush()
        except OSError:
            pass
    if not path:
        return {"path": None, "tail": "", "truncated": False}
    source = Path(str(path))
    limit = _tail_bytes()
    try:
        size = source.stat().st_size
        offset = max(0, size - limit)
        with source.open("rb") as stream:
            stream.seek(offset)
            tail = stream.read(limit).decode("utf-8", errors="replace")
    except OSError as exc:
        return {"path": str(source), "tail": "", "truncated": False, "read_error": str(exc)}
    return {"path": str(source), "tail": _redact(tail), "truncated": offset > 0}


def _safe_json(value: Any) -> Any:
    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        serialized = repr(value)
    try:
        return json.loads(_redact(serialized))
    except (TypeError, ValueError):
        return _redact(serialized)


def _emit_diagnostic(environment: Any, *, kind: str, action: Any = None, observation: Any = None) -> None:
    process = _field(environment, "_process")
    payload = {
        "kind": kind,
        "time": time(),
        "scenario": _field(environment, "_scenario"),
        "task_idx": _field(environment, "_task_idx"),
        "tool_name": _field(action, "tool_name"),
        "arguments": _safe_json(_field(action, "arguments")),
        "reward_type": _field(observation, "reward_type"),
        "error": _redact(str(_field(observation, "error", ""))),
        "verify_result": _safe_json(_field(observation, "verify_result")),
        "subprocess_log": _read_subprocess_log(process),
    }
    payload = json.loads(json.dumps(payload, ensure_ascii=False, default=repr, allow_nan=False))
    directory = _diagnostics_directory()
    if directory is not None:
        target = directory / f"awm-diagnostic-{uuid4().hex}.json"
        temporary = target.with_suffix(".tmp")
        try:
            payload["diagnostic_path"] = str(target)
            rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            temporary.write_text(rendered, encoding="utf-8")
            temporary.replace(target)
        except OSError as exc:
            payload["diagnostic_write_error"] = str(exc)
    # One physical log line keeps the outer AWM log mirror atomic across
    # concurrent scenario sessions. SwanLab then chunks it for display while
    # preserving the escaped subprocess traceback and structured fields.
    _LOGGER.error(
        "NOISE_RL_AWM_SUBPROCESS_DIAGNOSTIC %s",
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=repr),
    )


def install_awm_server_diagnostics() -> None:
    """Patch OpenEnv's AWM methods once in the server process only."""
    module = importlib.import_module("agent_world_model_env.server.awm_environment")
    environment_type = module.AWMEnvironment
    if getattr(environment_type, "_noise_rl_diagnostics_installed", False):
        return
    original_tool = environment_type._handle_call_tool
    original_verify = environment_type._handle_verify

    def handle_tool(self, action, timeout_s=None):
        observation = original_tool(self, action, timeout_s)
        if _field(observation, "reward_type") == "server_error":
            _emit_diagnostic(self, kind="tool_server_error", action=action, observation=observation)
        return observation

    def handle_verify(self, action):
        observation = original_verify(self, action)
        if _field(observation, "reward_type") not in {"complete", "incomplete"}:
            _emit_diagnostic(self, kind="unexpected_verifier_outcome", action=action, observation=observation)
        return observation

    environment_type._handle_call_tool = handle_tool
    environment_type._handle_verify = handle_verify
    environment_type._noise_rl_diagnostics_installed = True
    _LOGGER.info("Installed project-local AWM subprocess diagnostic hooks")
