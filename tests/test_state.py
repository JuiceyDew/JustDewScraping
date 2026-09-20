"""Runtime state and secrets: the settings.json the web UI writes."""

from __future__ import annotations

import json
import os
import stat

import pytest

from ideafindr import state


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("IDEAFINDR_STATE_DIR", str(tmp_path))
    return tmp_path


def test_save_is_0600_and_roundtrips(state_dir):
    state.save_overrides({"ollama_api_key": "sk-abcdef123456", "fast_model": "gpt-oss:120b"})
    path = state_dir / "settings.json"
    assert path.exists()
    # Group/other must have no access: the file holds live cookies and an API key.
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert state.load_overrides() == {
        "ollama_api_key": "sk-abcdef123456",
        "fast_model": "gpt-oss:120b",
    }


def test_non_editable_keys_are_dropped(state_dir):
    """A form post or a bad edit must not be able to rewrite db_path etc."""
    state.save_overrides({"db_path": "/etc/passwd", "ollama_api_key": "k", "nonsense": 1})
    stored = state.load_overrides()
    assert "db_path" not in stored and "nonsense" not in stored
    assert stored["ollama_api_key"] == "k"


def test_corrupt_file_degrades_to_defaults(state_dir):
    (state_dir / "settings.json").write_text("{not json", encoding="utf-8")
    assert state.load_overrides() == {}


def test_mask_never_reveals_the_middle():
    assert state.mask("sk-abcdef123456") == "sk-********56"
    assert state.mask("short") == "*****"
    assert state.mask("") == ""
    assert "abcdef" not in state.mask("sk-abcdef123456")


def test_secret_is_set_ignores_whitespace():
    assert state.secret_is_set({"k": "  "}, "k") is False
    assert state.secret_is_set({"k": " x "}, "k") is True
    assert state.secret_is_set({}, "k") is False


def test_save_merges_rather_than_replaces(state_dir):
    state.save_overrides({"fast_model": "a"})
    state.save_overrides({"smart_model": "b"})
    stored = state.load_overrides()
    assert stored["fast_model"] == "a" and stored["smart_model"] == "b"


def test_apply_overrides_respects_environment(state_dir, monkeypatch):
    """A Nix EnvironmentFile must win over whatever the UI stored."""
    state.save_overrides({"fast_model": "from-ui"})
    monkeypatch.setenv("FAST_MODEL", "from-env")

    class Fake:
        fast_model = "default"

    f = Fake()
    state.apply_overrides(f)
    # FAST_MODEL is set in the environment, so the stored value is skipped.
    assert f.fast_model == "default"

    monkeypatch.delenv("FAST_MODEL")
    state.apply_overrides(f)
    assert f.fast_model == "from-ui"
