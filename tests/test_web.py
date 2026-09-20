"""Web UI routes and the job registry, exercised with FastAPI's TestClient."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import ideafindr.config as cfg
from ideafindr.web import app as webapp


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate both the database and the settings file per test.
    monkeypatch.setenv("IDEAFINDR_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg.settings, "db_path", tmp_path / "t.db")
    monkeypatch.setattr(cfg.settings, "reports_dir", tmp_path / "reports")
    return TestClient(webapp.app)


def test_pages_render(client):
    for path in ("/", "/settings", "/credentials"):
        r = client.get(path)
        assert r.status_code == 200
        assert "ideafindr" in r.text


def test_health_reports_latest_run(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_unknown_job_is_404(client):
    assert client.get("/research/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_unknown_run_is_404(client):
    assert client.get("/runs/nope").status_code == 404


def test_settings_do_not_leak_secrets(client, monkeypatch):
    """The API key must never be echoed back to the page, only a masked placeholder."""
    from ideafindr.state import save_overrides

    save_overrides({"ollama_api_key": "sk-supersecret-value"})
    r = client.get("/settings")
    assert "sk-supersecret-value" not in r.text
    assert "(set)" in r.text


def test_settings_blank_secret_does_not_wipe_stored_value(client):
    from ideafindr.state import load_overrides, save_overrides

    save_overrides({"ollama_api_key": "sk-keep-me"})
    # A form submit with the secret field left blank (as rendered) must preserve it.
    r = client.post("/settings", data={"fast_model": "gpt-oss:120b"}, follow_redirects=False)
    assert r.status_code == 303
    assert load_overrides()["ollama_api_key"] == "sk-keep-me"


def test_credentials_x_is_stored(client):
    from ideafindr.state import load_overrides

    r = client.post("/credentials/x",
                    data={"auth_token": "tok-123", "ct0": "csrf-xyz"},
                    follow_redirects=False)
    assert r.status_code == 303
    stored = load_overrides()
    assert stored["x_auth_token"] == "tok-123" and stored["x_ct0"] == "csrf-xyz"


def test_x_paste_header_is_parsed(client):
    """The one-paste path: the whole Cookie header, HttpOnly values included."""
    from ideafindr.state import load_overrides

    r = client.post("/credentials/x/paste",
                    data={"cookies": "auth_token=tok-abc; ct0=csrf-def"},
                    follow_redirects=False)
    assert r.status_code == 303
    stored = load_overrides()
    assert stored["x_auth_token"] == "tok-abc" and stored["x_ct0"] == "csrf-def"


def test_x_paste_without_token_reports_error(client):
    r = client.post("/credentials/x/paste", data={"cookies": "ct0=only-csrf"},
                    follow_redirects=False)
    assert r.status_code == 303
    assert "error=" in r.headers["location"]


def test_instagram_paste_header_is_parsed(client):
    from ideafindr.state import load_overrides

    r = client.post("/credentials/instagram/paste",
                    data={"cookies": "sessionid=ig-123; csrftoken=ig-csrf"},
                    follow_redirects=False)
    assert r.status_code == 303
    stored = load_overrides()
    assert stored["instagram_sessionid"] == "ig-123"
    assert stored["instagram_csrftoken"] == "ig-csrf"


def test_import_without_browser_reports_error_not_500(client, monkeypatch):
    """A failed cookie import must be a friendly redirect, never a crash."""
    monkeypatch.setattr(webapp.browsercookies, "read_cookies", lambda p: {})
    r = client.post("/credentials/import/x", follow_redirects=False)
    assert r.status_code == 303
    assert "error=" in r.headers["location"]


def test_import_unsupported_platform_is_400(client):
    assert client.post("/credentials/import/tiktok").status_code == 400


def test_research_rejects_empty_topic(client):
    r = client.post("/research", data={"topic": "   "})
    assert r.status_code == 400


def _wait(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["finished"]:
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_full_research_flow_renders_a_report(client, monkeypatch):
    """Home -> start -> stages complete -> run page -> report served."""
    from datetime import datetime, timezone

    import ideafindr.pipeline as pipeline
    from ideafindr.models import Document

    async def fake_collect(plan, sources, limit=600):
        texts = (["chiller noise is awful at night"] * 8
                 + ["water gets cloudy after a week"] * 8
                 + ["running cost is higher than advertised"] * 8)
        return [
            Document(
                id=f"reddit:{i}", platform="reddit", kind="comment",
                url=f"https://reddit.com/{i}", text=t,
                created_at=datetime.now(timezone.utc), community="coldplunge",
                engagement={"score": i},
            )
            for i, t in enumerate(texts)
        ]

    monkeypatch.setattr(pipeline, "collect", fake_collect)
    # Keep the test offline: no LLM key means run_analysis keeps keyword labels
    # and never touches Ollama.
    monkeypatch.setattr(cfg.settings, "ollama_api_key", "")
    # The fake corpus has three texts, so k-means' auto-k may ask for more
    # clusters than there are distinct points; that warning is expected here.
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    r = client.post("/research",
                    data={"topic": "cold plunge", "days": "90", "sources": ["reddit"]},
                    follow_redirects=False)
    assert r.status_code == 303
    job_id = r.headers["location"].rsplit("/", 1)[-1]
    job = _wait(client, job_id)
    assert job["status"] == "done", job["error"]
    assert all(s == "done" for s in job["stage_states"].values())

    run_id = job["run_id"]
    page = client.get(f"/runs/{run_id}")
    assert page.status_code == 200
    report = client.get(f"/runs/{run_id}/report")
    assert report.status_code == 200
    assert "text/html" in report.headers["content-type"]
