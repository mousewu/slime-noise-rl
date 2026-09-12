import json
import logging

from noise_rl.awm_log_mirror import AWMServerLogMirror


class Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []
        self.levels = []

    def emit(self, record):
        self.messages.append(record.getMessage())
        self.levels.append(record.levelname)


def _logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    records = Records()
    logger.addHandler(records)
    return logger, records


def test_awm_mirror_forwards_server_error_context_and_redacts_secret(tmp_path, monkeypatch):
    server_log = tmp_path / "awm-server.log"
    server_log.write_text(
        "request scenario=crm task=2\n"
        "calling create_skill\n"
        "ERROR: Error calling create_skill. Status code: 500\n"
        "Traceback (most recent call last):\n"
        "Authorization: Bearer private-key\n"
        "RuntimeError: missing database table\n"
    )
    monkeypatch.setenv("SWANLAB_API_KEY", "private-key")
    logger, records = _logger("test.awm_log_mirror")
    audit = tmp_path / "run" / "awm_server_diagnostics.log"
    mirror = AWMServerLogMirror(server_log, audit_path=audit, followup_lines=3, logger=logger)

    mirror.poll_once()

    assert any("calling create_skill" in message for message in records.messages)
    assert any("Status code: 500" in message for message in records.messages)
    assert any("AWM server diagnostic context" in message for message in records.messages)
    assert records.levels and set(records.levels) == {"ERROR"}
    assert "private-key" not in audit.read_text()
    assert "Authorization: Bearer <REDACTED>" in audit.read_text()


def test_awm_mirror_waits_for_server_log_and_ignores_benign_lines(tmp_path):
    server_log = tmp_path / "late-server.log"
    logger, records = _logger("test.awm_log_mirror_late")
    mirror = AWMServerLogMirror(server_log, logger=logger)

    mirror.poll_once()
    server_log.write_text("INFO: environment ready\nINFO: request complete\n")
    mirror.poll_once()

    assert records.messages == []


def test_awm_mirror_reports_compact_verifier_evidence_metrics(tmp_path):
    server_log = tmp_path / "awm-server.log"
    server_log.write_text(
        "NOISE_RL_AWM_VERIFIER_EVIDENCE "
        + json.dumps(
            {
                "kind": "code_verifier_noncomplete",
                "scenario": "threading_demo",
                "task_idx": 3,
                "reward_type": "others",
                "evidence": {
                    "status": "saved",
                    "path": "/tmp/evidence.json",
                    "db_backups_saved": 2,
                    "changed_tables": 1,
                    "trajectory_entries": 7,
                },
            }
        )
        + "\n"
    )
    forwarded = []
    logger, records = _logger("test.awm_log_mirror_verifier_metrics")
    mirror = AWMServerLogMirror(server_log, logger=logger, metric_reporter=forwarded.append)

    mirror.poll_once()
    mirror.stop()

    assert len(forwarded) == 1
    assert forwarded[0] == {
        "awm/verifier/noncomplete/total": 1.0,
        "awm/verifier/noncomplete/unique_tasks": 1.0,
        "awm/verifier/others/total": 1.0,
        "awm/verifier/evidence/saved_total": 1.0,
        "awm/verifier/evidence/skipped_total": 0.0,
        "awm/verifier/evidence/disabled_total": 0.0,
        "awm/verifier/evidence/error_total": 0.0,
        "awm/verifier/evidence/db_backups_saved_total": 2.0,
        "awm/server/tool_server_error/total": 0.0,
        "awm/server/tool_server_error/unique_tasks": 0.0,
    }
    assert any("scenario=threading_demo task_idx=3" in message for message in records.messages)


def test_awm_mirror_counts_structured_tool_server_errors(tmp_path):
    server_log = tmp_path / "awm-server.log"
    server_log.write_text(
        "NOISE_RL_AWM_SUBPROCESS_DIAGNOSTIC "
        + json.dumps(
            {
                "kind": "tool_server_error",
                "scenario": "demo",
                "task_idx": 4,
                "tool_name": "write_record",
                "diagnostic_path": "/tmp/diagnostic.json",
            }
        )
        + "\n"
    )
    forwarded = []
    logger, records = _logger("test.awm_log_mirror_server_metrics")
    mirror = AWMServerLogMirror(server_log, logger=logger, metric_reporter=forwarded.append)

    mirror.poll_once()
    mirror.stop()

    assert forwarded[-1]["awm/server/tool_server_error/total"] == 1
    assert forwarded[-1]["awm/server/tool_server_error/unique_tasks"] == 1
    assert any("tool=write_record" in message for message in records.messages)
