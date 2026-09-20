"""Credential health probes.

A dead cookie must read as "expired", not "no data", and a missing optional
dependency must degrade to a warning rather than crash doctor. These tests pin
both, and never touch the network.
"""

from __future__ import annotations

import pytest

import ideafindr.config as cfg
from ideafindr.collectors import health


def test_x_unconfigured_is_a_status_not_an_error(monkeypatch):
    monkeypatch.setattr(cfg.settings, "x_auth_token", "")
    monkeypatch.setattr(cfg.settings, "x_ct0", "")
    ok, detail = health.probe_x()
    assert ok is False and "not configured" in detail


def test_instagram_probe_handles_missing_instaloader(monkeypatch):
    """The scrapers extra is optional; its absence must not raise."""
    monkeypatch.setattr(cfg.settings, "instagram_sessionid", "x")
    ok, detail = health.probe_instagram()
    # Either instaloader is absent (warn) or present but the session is dead;
    # both are False and neither raised.
    assert ok is False
    assert isinstance(detail, str) and detail


def test_reddit_reports_transport(monkeypatch):
    monkeypatch.setattr(cfg.settings, "reddit_client_id", "")
    monkeypatch.setattr(cfg.settings, "reddit_client_secret", "")
    ok, detail = health.probe_reddit()
    assert ok is False and "Arctic Shift" in detail

    monkeypatch.setattr(cfg.settings, "reddit_client_id", "id")
    monkeypatch.setattr(cfg.settings, "reddit_client_secret", "secret")
    ok, detail = health.probe_reddit()
    assert ok is True and "official" in detail


def test_x_probe_reports_expired_session(monkeypatch):
    """A 401 from X must be named as an expired cookie, not a generic failure."""
    import sys
    import types

    import ideafindr.collectors.x_twitter as xt

    monkeypatch.setattr(cfg.settings, "x_auth_token", "tok")
    monkeypatch.setattr(cfg.settings, "x_ct0", "csrf")

    async def fake_ensure(api):
        return True

    class FakeAPI:
        def __init__(self, *a, **k):
            pass

        def search(self, q, limit=1):
            return object()  # an async generator placeholder; gather() decides

    async def fake_gather(_awaitable):
        raise RuntimeError("HTTP 401 Unauthorized")

    monkeypatch.setattr(xt, "_ensure_account", fake_ensure)
    monkeypatch.setitem(
        sys.modules, "twscrape",
        types.SimpleNamespace(API=FakeAPI, gather=fake_gather),
    )

    ok, detail = health.probe_x()
    assert ok is False and "expired" in detail.lower()

