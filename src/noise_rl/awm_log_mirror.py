"""Mirror actionable AWM-server diagnostics into the training log.

The AWM environment runs in a separate Python process so its Uvicorn/MCP
tracebacks are not Ray worker output. Tail one explicitly configured server
log and re-emit only error blocks through Python logging. The regular
SwanLab logging bridge then captures them without importing OpenEnv into the
Megatron environment.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import deque
from pathlib import Path
from time import monotonic


_LOGGER = logging.getLogger(__name__)
_ACTIONABLE_PATTERN = re.compile(
    r"(?:\b(?:fatal|critical|error|exception|traceback)\b|"
    r"\bstatus code:\s*5\d\d\b|\bhttp\s*5\d\d\b|"
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b)",
    re.IGNORECASE,
)
_AUTHORIZATION_PATTERN = re.compile(r"(authorization\s*[:=]\s*bearer\s+)\S+", re.IGNORECASE)
_MAX_INITIAL_BYTES = 64 * 1024
_MAX_READ_BYTES = 2 * 1024 * 1024
_DEDUP_SECONDS = 60.0
_MAX_DEDUP_ENTRIES = 1024
_DEFAULT_CONTEXT_LINES = 12
_DEFAULT_FOLLOWUP_LINES = 20


def _redact(value: str) -> str:
    for name in ("SWANLAB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENENV_AWM_LLM_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, f"<{name}_REDACTED>")
    return _AUTHORIZATION_PATTERN.sub(r"\1<REDACTED>", value)


def _environment_number(name: str, default: float, *, integer: bool = False) -> float | int:
    """Read optional diagnostic tuning without allowing it to stop training."""
    raw = os.environ.get(name)
    if raw is None:
        return int(default) if integer else default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        _LOGGER.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return int(default) if integer else default
    if value <= 0:
        _LOGGER.warning("Ignoring nonpositive %s=%r; using %s", name, raw, default)
        return int(default) if integer else default
    return value


class AWMServerLogMirror:
    """Incrementally tail one AWM server log without delaying rollout work."""

    def __init__(
        self,
        log_path: Path,
        *,
        audit_path: Path | None = None,
        poll_seconds: float = 1.0,
        context_lines: int = _DEFAULT_CONTEXT_LINES,
        followup_lines: int = _DEFAULT_FOLLOWUP_LINES,
        logger: logging.Logger | None = None,
    ):
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if context_lines < 0 or followup_lines < 0:
            raise ValueError("context line counts must be nonnegative")
        self.log_path = log_path
        self.audit_path = audit_path
        self.poll_seconds = poll_seconds
        self.context = deque(maxlen=context_lines)
        self.followup_lines = followup_lines
        self.logger = logger or _LOGGER
        self._offset: int | None = None
        self._followups = 0
        self._recent_errors: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="noise-rl-awm-log-mirror", daemon=True)
        self._thread.start()
        self.logger.info("AWM server diagnostics are mirrored from %s", self.log_path)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_seconds * 2))
            self._thread = None

    def poll_once(self) -> None:
        """Read newly appended server-log lines once; public for tests/final flush."""
        try:
            size = self.log_path.stat().st_size
        except OSError:
            return
        offset = self._offset
        if offset is None:
            offset = max(0, size - _MAX_INITIAL_BYTES)
        elif size < offset:
            offset = 0
        if size <= offset:
            self._offset = offset
            return
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                content = stream.read(_MAX_READ_BYTES)
                self._offset = stream.tell()
        except OSError:
            return
        for line in content.splitlines():
            self._handle_line(line)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # pragma: no cover - diagnostics must never stop training
                self.logger.exception("AWM server diagnostic mirror failed; continuing without this poll")
            self._stop.wait(self.poll_seconds)

    def _handle_line(self, line: str) -> None:
        line = _redact(line)
        # The server cannot emit this project-only tag. It prevents accidental
        # self-feeding if a user redirects driver output into AWM_SERVER_LOG.
        if "noise_rl.awm_log_mirror" in line or "[AWM server diagnostic" in line:
            return
        if self._followups:
            self._emit(line, followup=True)
            self._followups -= 1
        if _ACTIONABLE_PATTERN.search(line) and not self._is_duplicate(line):
            if self.context:
                self._emit("\n".join(self.context), context=True)
            self._emit(line)
            self._followups = self.followup_lines
        self.context.append(line)

    def _is_duplicate(self, line: str) -> bool:
        now = monotonic()
        previous = self._recent_errors.get(line)
        self._recent_errors[line] = now
        if len(self._recent_errors) > _MAX_DEDUP_ENTRIES:
            self._recent_errors = {
                value: seen for value, seen in self._recent_errors.items() if now - seen < _DEDUP_SECONDS
            }
        return previous is not None and now - previous < _DEDUP_SECONDS

    def _emit(self, message: str, *, context: bool = False, followup: bool = False) -> None:
        prefix = "AWM server diagnostic context" if context or followup else "AWM server diagnostic"
        rendered = f"[{prefix}][{self.log_path.name}] {message}"
        # SwanLab's regular mirror defaults to WARNING+. Keep the bounded
        # context and traceback at ERROR too, otherwise users would see only
        # a bare HTTP 500 rather than the server-side exception that explains
        # it. This is an error block, not routine server output.
        self.logger.error("%s", rendered)
        if self.audit_path is not None:
            try:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(rendered + "\n")
            except OSError:
                # Standard logging remains available even if the run directory
                # becomes read-only or temporarily unavailable.
                pass


def start_awm_server_log_mirror(*, audit_path: str | None = None) -> AWMServerLogMirror | None:
    """Start AWM server-log mirroring when the launcher supplied a log path."""
    if os.environ.get("NOISE_RL_AWM_LOG_MIRROR", "1").lower() in {"0", "false", "no"}:
        _LOGGER.info("AWM server diagnostic mirror is disabled by NOISE_RL_AWM_LOG_MIRROR")
        return None
    configured = os.environ.get("NOISE_RL_AWM_SERVER_LOG")
    if not configured:
        _LOGGER.info("AWM server diagnostic mirror is unavailable: NOISE_RL_AWM_SERVER_LOG is not set")
        return None
    poll_seconds = float(_environment_number("NOISE_RL_AWM_LOG_POLL_SECONDS", 1.0))
    context_lines = int(_environment_number("NOISE_RL_AWM_LOG_CONTEXT_LINES", _DEFAULT_CONTEXT_LINES, integer=True))
    followup_lines = int(
        _environment_number("NOISE_RL_AWM_LOG_FOLLOWUP_LINES", _DEFAULT_FOLLOWUP_LINES, integer=True)
    )
    mirror = AWMServerLogMirror(
        Path(configured).expanduser(),
        audit_path=Path(audit_path).expanduser() if audit_path else None,
        poll_seconds=poll_seconds,
        context_lines=context_lines,
        followup_lines=followup_lines,
    )
    mirror.start()
    return mirror
