"""Runtime diagnostics for OpenEnv's process-isolated AWM scenarios.

OpenEnv converts sub-environment MCP exceptions into ``server_error``
observations.  The original FastAPI traceback lives only in a per-session
``server.log`` which is normally cleaned up as soon as the WebSocket session
closes.  This module patches the running AWM server in memory, before Uvicorn
loads its app, and emits a bounded structured record while that file still
exists.

AWM's code verifier also deliberately emits ``others`` for a verifier that
ran successfully but did not pass.  That status alone cannot distinguish a
policy failure from an environment or verifier bug.  For a bounded sample of
such outcomes this module saves a self-contained evidence bundle: task and
verifier source, MCP trajectory, read-only SQLite snapshots and a table-level
database comparison.  It deliberately has only standard-library dependencies
so it can run inside the isolated NumPy-2 AWM environment.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import shutil
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from threading import Lock
from time import time
from typing import Any
from uuid import uuid4


_LOGGER = logging.getLogger(__name__)
_DEFAULT_LOG_TAIL_BYTES = 24 * 1024
_MAX_LOG_TAIL_BYTES = 256 * 1024
_DEFAULT_OTHERS_LOG_TAIL_BYTES = 4 * 1024
_DEFAULT_EVIDENCE_MAX_BUNDLES = 64
_DEFAULT_EVIDENCE_PER_TASK = 1
_DEFAULT_EVIDENCE_MAX_DB_BYTES = 16 * 1024 * 1024
_MAX_EVIDENCE_DB_BYTES = 256 * 1024 * 1024
_DEFAULT_EVIDENCE_MAX_TRAJECTORY_ENTRIES = 128
_DEFAULT_EVIDENCE_MAX_TRAJECTORY_BYTES = 512 * 1024
_DEFAULT_EVIDENCE_MAX_VERIFIER_BYTES = 256 * 1024
_DEFAULT_EVIDENCE_MAX_TASK_BYTES = 64 * 1024
_DEFAULT_EVIDENCE_MAX_DB_ROWS = 10_000
_DEFAULT_EVIDENCE_DB_SAMPLE_ROWS = 8
_SECRET_NAMES = (
    "SWANLAB_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENENV_AWM_LLM_API_KEY",
)


class _EvidenceBudget:
    """Bound diagnostic disk usage without dropping the aggregate event count."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._captured_total = 0
        self._captured_by_task: Counter[tuple[str, int | None]] = Counter()

    def claim(self, scenario: Any, task_idx: Any) -> tuple[bool, str]:
        maximum = _environment_integer(
            "NOISE_RL_AWM_EVIDENCE_MAX_BUNDLES", _DEFAULT_EVIDENCE_MAX_BUNDLES, minimum=0
        )
        if maximum == 0:
            return False, "disabled"
        per_task = _environment_integer(
            "NOISE_RL_AWM_EVIDENCE_PER_TASK", _DEFAULT_EVIDENCE_PER_TASK, minimum=1
        )
        key = (str(scenario), task_idx if isinstance(task_idx, int) else None)
        with self._lock:
            if self._captured_total >= maximum:
                return False, "global_limit"
            if self._captured_by_task[key] >= per_task:
                return False, "per_task_limit"
            self._captured_total += 1
            self._captured_by_task[key] += 1
        return True, "claimed"


_EVIDENCE_BUDGET = _EvidenceBudget()


def _diagnostics_directory() -> Path | None:
    configured = os.environ.get("NOISE_RL_AWM_DIAGNOSTICS_DIR")
    if not configured:
        return None
    directory = Path(configured).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _environment_integer(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    """Read a bounded integer option without allowing diagnostics to fail AWM."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        _LOGGER.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default
    if value < minimum:
        _LOGGER.warning("Ignoring %s=%r below minimum %d; using %d", name, raw, minimum, default)
        return default
    if maximum is not None:
        return min(value, maximum)
    return value


def _tail_bytes() -> int:
    return _environment_integer(
        "NOISE_RL_AWM_SUBPROCESS_LOG_TAIL_BYTES",
        _DEFAULT_LOG_TAIL_BYTES,
        minimum=0,
        maximum=_MAX_LOG_TAIL_BYTES,
    )


def _others_log_tail_bytes() -> int:
    return _environment_integer(
        "NOISE_RL_AWM_OTHERS_LOG_TAIL_BYTES",
        _DEFAULT_OTHERS_LOG_TAIL_BYTES,
        minimum=0,
        maximum=_MAX_LOG_TAIL_BYTES,
    )


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


def _read_subprocess_log(process: Any, *, limit: int | None = None) -> dict[str, Any]:
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
    limit = _tail_bytes() if limit is None else limit
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


def _sha256(value: str | bytes) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _bounded_text(value: Any, *, maximum: int) -> dict[str, Any]:
    """Return redacted text plus an explicit truncation boundary."""
    text = _redact(str(value or ""))
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum:
        return {"text": text, "bytes": len(encoded), "truncated": False}
    prefix = encoded[:maximum].decode("utf-8", errors="ignore")
    return {"text": prefix, "bytes": len(encoded), "truncated": True}


def _bounded_json_list(value: Any, *, maximum_entries: int, maximum_bytes: int) -> dict[str, Any]:
    """Preserve complete JSON entries until the evidence budget is exhausted."""
    normalized = _safe_json(value)
    if not isinstance(normalized, list):
        return {"entries": [], "entry_count": 0, "captured_entries": 0, "truncated": False}
    entries: list[Any] = []
    consumed = 2
    for item in normalized[:maximum_entries]:
        rendered = json.dumps(item, ensure_ascii=False, default=repr, allow_nan=False).encode("utf-8")
        if entries and consumed + len(rendered) + 1 > maximum_bytes:
            break
        if not entries and len(rendered) > maximum_bytes:
            break
        entries.append(item)
        consumed += len(rendered) + 1
    return {
        "entries": entries,
        "entry_count": len(normalized),
        "captured_entries": len(entries),
        "truncated": len(entries) < len(normalized),
        "max_entries": maximum_entries,
        "max_bytes": maximum_bytes,
    }


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"type": "bytes", "bytes": len(value), "sha256": _sha256(value)}
    return value


def _sqlite_database_summary(path: Any) -> dict[str, Any]:
    """Make a bounded, deterministic summary suitable for initial/final diffing."""
    result: dict[str, Any] = {"source_path": str(path) if path else None, "tables": []}
    if not path:
        result["error"] = "database path is unavailable"
        return result
    source = Path(str(path))
    try:
        source_stat = source.stat()
    except OSError as exc:
        result["error"] = str(exc)
        return result
    result["bytes"] = source_stat.st_size
    maximum_rows = _environment_integer(
        "NOISE_RL_AWM_EVIDENCE_MAX_DB_ROWS", _DEFAULT_EVIDENCE_MAX_DB_ROWS, minimum=1
    )
    sample_rows = _environment_integer(
        "NOISE_RL_AWM_EVIDENCE_DB_SAMPLE_ROWS", _DEFAULT_EVIDENCE_DB_SAMPLE_ROWS, minimum=0
    )
    try:
        with closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            tables = connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for table_name, schema in tables:
                quoted = _quote_identifier(str(table_name))
                columns = [row[1] for row in connection.execute(f"PRAGMA table_info({quoted})")]
                row_count = int(connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
                table: dict[str, Any] = {
                    "name": table_name,
                    "row_count": row_count,
                    "columns": columns,
                    "schema_sha256": _sha256(schema or ""),
                    "content_complete": row_count <= maximum_rows,
                }
                if not columns:
                    result["tables"].append(table)
                    continue
                ordering = ", ".join(_quote_identifier(str(column)) for column in columns)
                rows = connection.execute(
                    f"SELECT * FROM {quoted} ORDER BY {ordering} LIMIT ?", (maximum_rows + 1,)
                ).fetchall()
                if len(rows) > maximum_rows:
                    rows = rows[:maximum_rows]
                    table["content_complete"] = False
                digest = hashlib.sha256()
                normalized_rows = []
                for row in rows:
                    normalized = [_sqlite_value(value) for value in row]
                    normalized_rows.append(normalized)
                    digest.update(
                        json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=repr).encode("utf-8")
                    )
                    digest.update(b"\n")
                table["content_sha256"] = digest.hexdigest()
                table["sample_rows"] = normalized_rows[:sample_rows]
                result["tables"].append(table)
    except (OSError, sqlite3.Error, ValueError) as exc:
        result["error"] = str(exc)
    return result


def _sqlite_diff(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    """Compare table summaries; raw database backups remain authoritative."""
    initial_tables = {table["name"]: table for table in initial.get("tables", [])}
    final_tables = {table["name"]: table for table in final.get("tables", [])}
    changes = []
    for name in sorted(set(initial_tables) | set(final_tables)):
        before, after = initial_tables.get(name), final_tables.get(name)
        changed = before != after
        changes.append(
            {
                "table": name,
                "changed": changed,
                "initial_row_count": before.get("row_count") if before else None,
                "final_row_count": after.get("row_count") if after else None,
                "initial_content_sha256": before.get("content_sha256") if before else None,
                "final_content_sha256": after.get("content_sha256") if after else None,
                "initial_content_complete": before.get("content_complete") if before else None,
                "final_content_complete": after.get("content_complete") if after else None,
            }
        )
    return {"changed_tables": sum(item["changed"] for item in changes), "tables": changes}


def _backup_sqlite_database(source_path: Any, destination: Path) -> dict[str, Any]:
    """Use SQLite's backup API so WAL state is captured in a standalone file."""
    metadata: dict[str, Any] = {"source_path": str(source_path) if source_path else None}
    if not source_path:
        metadata["status"] = "unavailable"
        return metadata
    source = Path(str(source_path))
    try:
        source_size = source.stat().st_size
        wal = source.with_name(source.name + "-wal")
        wal_size = wal.stat().st_size if wal.exists() else 0
    except OSError as exc:
        metadata.update(status="unavailable", error=str(exc))
        return metadata
    maximum = _environment_integer(
        "NOISE_RL_AWM_EVIDENCE_MAX_DB_BYTES",
        _DEFAULT_EVIDENCE_MAX_DB_BYTES,
        minimum=0,
        maximum=_MAX_EVIDENCE_DB_BYTES,
    )
    metadata["source_bytes"] = source_size + wal_size
    if maximum == 0:
        metadata["status"] = "disabled"
        return metadata
    if source_size + wal_size > maximum:
        metadata.update(status="skipped_too_large", maximum_bytes=maximum)
        return metadata
    try:
        with closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as reader:
            with closing(sqlite3.connect(destination)) as writer:
                reader.backup(writer)
        destination_size = destination.stat().st_size
        if destination_size > maximum:
            destination.unlink(missing_ok=True)
            metadata.update(status="skipped_too_large", maximum_bytes=maximum)
            return metadata
        metadata.update(
            status="saved",
            # The containing temporary directory is atomically renamed after
            # capture, so paths stored in evidence are deliberately relative.
            path=destination.name,
            bytes=destination_size,
            sha256=_sha256(destination.read_bytes()),
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        destination.unlink(missing_ok=True)
        metadata.update(status="error", error=str(exc))
    return metadata


def _verifier_snapshot(environment: Any) -> dict[str, Any]:
    scenario, task_idx = _field(environment, "_scenario"), _field(environment, "_task_idx")
    loader = _field(environment, "_data_loader")
    try:
        entry = loader.get_verifier(scenario, task_idx, "code")
    except Exception as exc:
        return {"mode": "code", "error": f"failed to load verifier: {exc}"}
    verification = entry.get("verification", {}) if isinstance(entry, dict) else {}
    code = verification.get("code", "") if isinstance(verification, dict) else ""
    source = _bounded_text(
        code,
        maximum=_environment_integer(
            "NOISE_RL_AWM_EVIDENCE_MAX_VERIFIER_BYTES", _DEFAULT_EVIDENCE_MAX_VERIFIER_BYTES, minimum=1
        ),
    )
    source["sha256"] = _sha256(str(code))
    return {
        "mode": "code",
        "source": source,
        "metadata": _safe_json({key: value for key, value in verification.items() if key != "code"}),
    }


def _capture_verifier_evidence(environment: Any, action: Any, observation: Any) -> dict[str, Any]:
    """Persist the inputs needed to distinguish policy, environment and verifier failures."""
    directory = _diagnostics_directory()
    scenario, task_idx = _field(environment, "_scenario"), _field(environment, "_task_idx")
    if directory is None:
        return {"status": "disabled", "reason": "NOISE_RL_AWM_DIAGNOSTICS_DIR is not set"}
    claimed, reason = _EVIDENCE_BUDGET.claim(scenario, task_idx)
    if not claimed:
        return {"status": "skipped", "reason": reason}
    identifier = uuid4().hex
    temporary = directory / f".awm-evidence-{identifier}.tmp"
    target = directory / f"awm-evidence-{identifier}"
    try:
        temporary.mkdir(parents=False, exist_ok=False)
        initial_path, final_path = _field(environment, "_initial_db_path"), _field(environment, "_db_path")
        initial_summary = _sqlite_database_summary(initial_path)
        final_summary = _sqlite_database_summary(final_path)
        evidence = {
            "schema_version": 1,
            "kind": "code_verifier_noncomplete",
            "captured_at": time(),
            "scenario": scenario,
            "task_idx": task_idx,
            "task": _bounded_text(
                _field(environment, "_task", ""),
                maximum=_environment_integer(
                    "NOISE_RL_AWM_EVIDENCE_MAX_TASK_BYTES", _DEFAULT_EVIDENCE_MAX_TASK_BYTES, minimum=1
                ),
            ),
            "verify_action": {
                "tool_name": _field(action, "tool_name"),
                "arguments": _safe_json(_field(action, "arguments")),
            },
            "observation": {
                "reward_type": _field(observation, "reward_type"),
                "error": _redact(str(_field(observation, "error", ""))),
                "verify_result": _safe_json(_field(observation, "verify_result")),
            },
            "verifier": _verifier_snapshot(environment),
            "trajectory": _bounded_json_list(
                _field(environment, "_trajectory", []),
                maximum_entries=_environment_integer(
                    "NOISE_RL_AWM_EVIDENCE_MAX_TRAJECTORY_ENTRIES",
                    _DEFAULT_EVIDENCE_MAX_TRAJECTORY_ENTRIES,
                    minimum=1,
                ),
                maximum_bytes=_environment_integer(
                    "NOISE_RL_AWM_EVIDENCE_MAX_TRAJECTORY_BYTES",
                    _DEFAULT_EVIDENCE_MAX_TRAJECTORY_BYTES,
                    minimum=1024,
                ),
            ),
            "tools": _bounded_json_list(
                _field(environment, "_tools_cache", []), maximum_entries=256, maximum_bytes=256 * 1024
            ),
            "database": {
                "initial": initial_summary,
                "final": final_summary,
                "diff": _sqlite_diff(initial_summary, final_summary),
            },
            "subprocess_log": _read_subprocess_log(
                _field(environment, "_process"), limit=_others_log_tail_bytes()
            ),
            "session_dir": _field(environment, "_session_dir"),
        }
        evidence["database"]["initial_backup"] = _backup_sqlite_database(
            initial_path, temporary / "initial.sqlite"
        )
        evidence["database"]["final_backup"] = _backup_sqlite_database(final_path, temporary / "final.sqlite")
        rendered = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, default=repr, allow_nan=False) + "\n"
        (temporary / "evidence.json").write_text(rendered, encoding="utf-8")
        temporary.replace(target)
        return {
            "status": "saved",
            "path": str(target / "evidence.json"),
            "db_backups_saved": sum(
                item.get("status") == "saved"
                for item in (evidence["database"]["initial_backup"], evidence["database"]["final_backup"])
            ),
            "changed_tables": evidence["database"]["diff"]["changed_tables"],
            "trajectory_entries": evidence["trajectory"]["captured_entries"],
        }
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        return {"status": "error", "error": str(exc)}


def _diagnostic_payload(environment: Any, *, kind: str, action: Any = None, observation: Any = None) -> dict[str, Any]:
    process = _field(environment, "_process")
    return {
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


def _persist_diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
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
    return payload


def _emit_diagnostic(
    environment: Any,
    *,
    kind: str,
    action: Any = None,
    observation: Any = None,
    level: int = logging.ERROR,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = _diagnostic_payload(environment, kind=kind, action=action, observation=observation)
    if extra:
        payload.update(extra)
    payload = _persist_diagnostic(payload)
    # One physical log line keeps the outer AWM log mirror atomic across
    # concurrent scenario sessions. SwanLab then chunks it for display while
    # preserving the escaped subprocess traceback and structured fields.
    _LOGGER.log(
        level,
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
        reward_type = _field(observation, "reward_type")
        if reward_type == "others":
            evidence = _capture_verifier_evidence(self, action, observation)
            # Keep the outer server log compact: detailed trajectories and DB
            # files live only in the bounded evidence bundle.  The trainer's
            # log mirror consumes this marker to publish aggregate SwanLab
            # counters without treating an ordinary non-pass as a server error.
            _LOGGER.info(
                "NOISE_RL_AWM_VERIFIER_EVIDENCE %s",
                json.dumps(
                    {
                        "kind": "code_verifier_noncomplete",
                        "scenario": _field(self, "_scenario"),
                        "task_idx": _field(self, "_task_idx"),
                        "reward_type": reward_type,
                        "evidence": _safe_json(evidence),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=repr,
                ),
            )
        elif reward_type not in {"complete", "incomplete"}:
            _emit_diagnostic(self, kind="unexpected_verifier_outcome", action=action, observation=observation)
        return observation

    environment_type._handle_call_tool = handle_tool
    environment_type._handle_verify = handle_verify
    environment_type._noise_rl_diagnostics_installed = True
    _LOGGER.info("Installed project-local AWM subprocess diagnostic hooks")
