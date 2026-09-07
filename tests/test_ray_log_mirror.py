import logging

from noise_rl.ray_log_mirror import RayLogMirror, find_ray_log_directory


class Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_find_ray_log_directory_uses_current_session(tmp_path):
    logs = tmp_path / "ray" / "session_abc" / "logs"
    logs.mkdir(parents=True)
    (tmp_path / "ray" / "session_latest").symlink_to(logs.parent)

    assert find_ray_log_directory(str(tmp_path / "ray")).resolve() == logs


def test_mirror_emits_error_with_context_and_writes_durable_audit(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    worker_log = logs / "worker-example.err"
    worker_log.write_text("engine ready\nrequest=42\nCUDA error: out of memory\nstack frame\n")
    monkeypatch.setenv("SWANLAB_API_KEY", "private-key")

    logger = logging.getLogger("test.ray_log_mirror")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    records = Records()
    logger.addHandler(records)
    audit = tmp_path / "run" / "ray_diagnostics.log"
    mirror = RayLogMirror(logs, audit_path=audit, followup_lines=2, logger=logger)

    mirror.poll_once()
    worker_log.write_text(worker_log.read_text() + "Bearer private-key\nnext frame\n")
    mirror.poll_once()

    assert any("request=42\nCUDA error" in message for message in records.messages)
    assert any("Ray diagnostic context" in message for message in records.messages)
    assert "private-key" not in audit.read_text()
    assert "<SWANLAB_API_KEY_REDACTED>" in audit.read_text()
