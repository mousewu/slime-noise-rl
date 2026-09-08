"""Mirror actionable Ray session diagnostics into the durable training log.

Ray's worker, raylet, and inference-server stderr normally lives only inside
the ephemeral Ray session directory.  ``log_to_driver`` is useful but does
not reliably preserve native-process output (notably SGLang engine failures).
This module tails the current local session and forwards error blocks through
the ordinary Python logger, which means they reach both the launcher log and
the optional SwanLab log mirror without modifying Slime.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from collections import defaultdict, deque
from pathlib import Path
from time import monotonic


_LOGGER = logging.getLogger(__name__)
_ERROR_PATTERN = re.compile(
    r"(?:\b(?:[a-z_]*error|[a-z_]*exception|fatal|abort(?:ed)?|sig(?:term|kill|segv|abrt))\b"
    r"|out of memory|cuda (?:error|exception)|segmentation fault"
    r"|(?:worker|driver).{0,80}\b(?:died|crashed|killed)\b)",
    re.IGNORECASE,
)
_AUTHORIZATION_PATTERN = re.compile(r"(authorization\s*[:=]\s*bearer\s+)\S+", re.IGNORECASE)
_MAX_INITIAL_BYTES = 64 * 1024
_MAX_READ_BYTES = 2 * 1024 * 1024
_DEDUP_SECONDS = 60.0
_MAX_DEDUP_ENTRIES = 2048


def find_ray_log_directory(ray_tmpdir: str | None) -> Path | None:
    """Return the active local Ray ``logs`` directory when it is discoverable."""
    candidates: list[Path] = []
    if ray_tmpdir:
        root = Path(ray_tmpdir)
        candidates.extend((root / "session_latest" / "logs",))

    # The private Node accessor is the most accurate source once Ray has
    # initialized.  Keep the fallback paths below so this remains compatible
    # with Ray versions where that accessor changes.
    try:
        from ray._private import worker

        node = getattr(worker, "_global_node", None)
        if node is not None:
            get_logs_dir = getattr(node, "get_logs_dir_path", None)
            if callable(get_logs_dir):
                candidates.append(Path(get_logs_dir()))
            get_session_dir = getattr(node, "get_session_dir_path", None)
            if callable(get_session_dir):
                candidates.append(Path(get_session_dir()) / "logs")
    except (ImportError, OSError, TypeError, ValueError):
        pass

    if not ray_tmpdir:
        candidates.append(Path(tempfile.gettempdir()) / "ray" / "session_latest" / "logs")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _redact(value: str) -> str:
    for name in ("SWANLAB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, f"<{name}_REDACTED>")
    return _AUTHORIZATION_PATTERN.sub(r"\1<REDACTED>", value)


class RayLogMirror:
    """Tail error blocks from one Ray session without delaying training work."""

    def __init__(
        self,
        log_dir: Path,
        *,
        audit_path: Path | None = None,
        poll_seconds: float = 1.0,
        context_lines: int = 20,
        followup_lines: int = 30,
        logger: logging.Logger | None = None,
    ):
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if context_lines < 0 or followup_lines < 0:
            raise ValueError("context line counts must be nonnegative")
        self.log_dir = log_dir
        self.audit_path = audit_path
        self.poll_seconds = poll_seconds
        self.context_lines = context_lines
        self.followup_lines = followup_lines
        self.logger = logger or _LOGGER
        self._offsets: dict[Path, int] = {}
        self._context: dict[Path, deque[str]] = defaultdict(lambda: deque(maxlen=context_lines))
        self._followups: dict[Path, int] = defaultdict(int)
        self._recent_errors: dict[tuple[str, str], float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="noise-rl-ray-log-mirror", daemon=True)
        self._thread.start()
        self.logger.info("Ray diagnostics are mirrored from %s", self.log_dir)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_seconds * 2))
            self._thread = None

    def poll_once(self) -> None:
        """Read new lines once; public to make the mirror straightforward to test."""
        for path in self._log_files():
            self._read_file(path)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # pragma: no cover - defensive: diagnostics must not stop training
                self.logger.exception("Ray diagnostic mirror failed; continuing without this poll")
            self._stop.wait(self.poll_seconds)

    def _log_files(self) -> list[Path]:
        try:
            return sorted(path for path in self.log_dir.iterdir() if path.is_file())
        except OSError:
            return []

    def _read_file(self, path: Path) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            return
        offset = self._offsets.get(path)
        if offset is None:
            offset = max(0, size - _MAX_INITIAL_BYTES)
        elif size < offset:
            offset = 0
        if size <= offset:
            self._offsets[path] = offset
            return
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                content = stream.read(_MAX_READ_BYTES)
                self._offsets[path] = stream.tell()
        except OSError:
            return
        for line in content.splitlines():
            self._handle_line(path, line)

    def _handle_line(self, path: Path, line: str) -> None:
        line = _redact(line)
        # SwanLabLogger writes forwarded records to its own Ray worker stdout.
        # Tailing that file again would feed this mirror's output back into
        # itself indefinitely. These tags can only have been emitted by this
        # project, never by an upstream Ray/SGLang diagnostic.
        if "noise_rl.ray_log_mirror" in line or "[Ray diagnostic" in line:
            return
        context = self._context[path]
        followups = self._followups[path]
        if followups:
            self._emit(path, line, followup=True)
            self._followups[path] = followups - 1
        if _ERROR_PATTERN.search(line) and not self._is_duplicate(path, line):
            if context:
                self._emit(path, "\n".join(context), context=True)
            self._emit(path, line)
            self._followups[path] = self.followup_lines
        context.append(line)

    def _is_duplicate(self, path: Path, line: str) -> bool:
        now = monotonic()
        key = (path.name, line)
        previous = self._recent_errors.get(key)
        self._recent_errors[key] = now
        if len(self._recent_errors) > _MAX_DEDUP_ENTRIES:
            self._recent_errors = {
                item: seen for item, seen in self._recent_errors.items() if now - seen < _DEDUP_SECONDS
            }
        return previous is not None and now - previous < _DEDUP_SECONDS

    def _emit(
        self,
        path: Path,
        message: str,
        *,
        context: bool = False,
        followup: bool = False,
    ) -> None:
        relative = path.name
        prefix = "Ray diagnostic context" if context or followup else "Ray diagnostic"
        rendered = f"[{prefix}][{relative}] {message}"
        if context or followup:
            self.logger.info("%s", rendered)
        else:
            self.logger.error("%s", rendered)
        if self.audit_path is not None:
            try:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(rendered + "\n")
            except OSError:
                # The primary training logger remains available even if the
                # output filesystem has a transient issue.
                pass


def start_ray_log_mirror(
    ray_tmpdir: str | None,
    *,
    audit_path: str | None = None,
) -> RayLogMirror | None:
    """Start local-session mirroring unless the user explicitly disables it."""
    if os.environ.get("NOISE_RL_RAY_LOG_MIRROR", "1").lower() in {"0", "false", "no"}:
        _LOGGER.info("Ray diagnostic mirror is disabled by NOISE_RL_RAY_LOG_MIRROR")
        return None
    log_dir = find_ray_log_directory(ray_tmpdir)
    if log_dir is None:
        _LOGGER.warning(
            "Ray diagnostic mirror could not find a local session log directory; "
            "Ray may be using a remote cluster"
        )
        return None
    interval = float(os.environ.get("NOISE_RL_RAY_LOG_POLL_SECONDS", "1"))
    mirror = RayLogMirror(
        log_dir,
        audit_path=Path(audit_path).expanduser() if audit_path else None,
        poll_seconds=interval,
    )
    mirror.start()
    return mirror
