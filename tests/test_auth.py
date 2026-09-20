"""The single-password login gate.

The server is headless and the UI holds an API key and live session cookies, so
whenever a password is set every route must be gated -- not just the obvious ones
-- and the session cookie must be signed and tamper-evident.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import ideafindr.config as cfg
from ideafindr.web import app as webapp
from ideafindr.web import auth


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IDEAFINDR_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg.settings, "db_path", tmp_path / "t.db")
    monkeypatch.setattr(cfg.settings, "reports_dir", tmp_path / "reports")
    return TestClient(webapp.app, follow_redirects=False)


def test_gate_disabled_by_default(client):
    """No password means the UI behaves exactly as before."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cfg.settings, "auth_password", "")
    assert auth.enabled() is False
    assert client.get("/").status_code == 200
    monkeypatch.undo()


def test_protected_routes_redirect_to_login(client, monkeypatch):
    monkeypatch.setattr(cfg.settings, "auth_password", "hunter2")
    for path in ("/", "/settings", "/credentials", "/runs/x"):
        r = client.get(path)
        assert r.status_code == 303, path
        assert r.headers["location"].startswith("/login"), path


def test_health_stays_open_for_monitoring(client, monkeypatch):
    monkeypatch.setattr(cfg.settings, "auth_password", "hunter2")
    assert client.get("/health").status_code == 200


def test_wrong_password_is_rejected(client, monkeypatch):
    monkeypatch.setattr(cfg.settings, "auth_password", "hunter2")
    r = client.post("/login", data={"password": "nope", "next": "/"})
    assert r.status_code == 303
    assert "error=1" in r.headers["location"]
    assert "ideafindr_session" not in r.cookies


def test_correct_password_sets_a_signed_cookie_and_unlocks(client, monkeypatch):
    monkeypatch.setattr(cfg.settings, "auth_password", "hunter2")
    r = client.post("/login", data={"password": "hunter2", "next": "/settings"})
    assert r.status_code == 303 and r.headers["location"] == "/settings"
    token = r.cookies["ideafindr_session"]
    assert auth.valid_token(token)
    # The cookie now satisfies the middleware.
    r2 = client.get("/settings")
    assert r2.status_code == 200


def test_tampered_cookie_is_rejected(client, monkeypatch):
    monkeypatch.setattr(cfg.settings, "auth_password", "hunter2")
    client.cookies.set("ideafindr_session", "not.a.valid.token")
    assert client.get("/").status_code == 303


def test_login_next_cannot_redirect_off_site(client, monkeypatch):
    """A crafted ?next= must not turn the login page into an open redirect."""
    monkeypatch.setattr(cfg.settings, "auth_password", "hunter2")
    r = client.post("/login",
                    data={"password": "hunter2", "next": "https://evil.example"})
    assert r.status_code == 303
    assert r.headers["location"] == "/"


def test_signature_depends_on_the_secret(client, monkeypatch):
    monkeypatch.setattr(cfg.settings, "auth_password", "hunter2")
    token = auth.make_token()
    assert auth.valid_token(token)
    # Flip the signature: must fail.
    b64, _, sig = token.partition(".")
    assert not auth.valid_token(f"{b64}.{sig[:-1]}0")


def test_expired_token_is_rejected(client, monkeypatch):
    monkeypatch.setattr(cfg.settings, "auth_password", "hunter2")
    import base64
    import hashlib
    import hmac

    old = base64.urlsafe_b64encode(str(int(time.time()) - auth.SESSION_TTL - 10).encode())
    sig = hmac.new(auth._secret(), base64.urlsafe_b64decode(old), hashlib.sha256).hexdigest()
    assert not auth.valid_token(f"{old.decode()}.{sig}")


def test_logout_clears_the_cookie(client, monkeypatch):
    monkeypatch.setattr(cfg.settings, "auth_password", "hunter2")
    r = client.post("/login", data={"password": "hunter2", "next": "/"})
    assert auth.valid_token(r.cookies["ideafindr_session"])
    r2 = client.post("/logout")
    assert r2.status_code == 303
    assert r2.headers["location"] == "/login"
