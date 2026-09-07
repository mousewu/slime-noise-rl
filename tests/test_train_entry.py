import pytest

from noise_rl import train_entry
from noise_rl.train_entry import ray_temp_directory


def test_ray_temp_directory_preserves_short_symlink_style_path(monkeypatch):
    short_path = "/tmp/nrl"
    monkeypatch.setenv("NOISE_RL_RAY_TMPDIR", short_path)
    monkeypatch.setenv("TMPDIR", "/a/deliberately/longer/fast-downward/path")
    monkeypatch.setattr(train_entry.Path, "mkdir", lambda *args, **kwargs: None)

    assert ray_temp_directory() == f"{short_path}/ray"


def test_ray_temp_directory_rejects_socket_path_overflow(monkeypatch):
    monkeypatch.setenv("NOISE_RL_RAY_TMPDIR", "/" + "a" * 80)
    monkeypatch.delenv("TMPDIR", raising=False)

    with pytest.raises(ValueError, match="too long"):
        ray_temp_directory()


def test_ray_temp_directory_returns_none_without_explicit_path(monkeypatch):
    monkeypatch.delenv("NOISE_RL_RAY_TMPDIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)

    assert ray_temp_directory() is None
