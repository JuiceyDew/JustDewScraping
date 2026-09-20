"""Minimal server-rendered web UI.

Server-rendered Jinja templates, a few lines of vanilla JS for progress polling,
one small stylesheet. No build step, no CDN, and no client-side secret handling:
all values that could be scraped or forged are escaped by Jinja's autoescape, and
secrets are never sent back to the browser.

This is a single-user tool meant for a LAN or VPN-reachable homelab -- there is
no app auth by design. It binds 0.0.0.0 (web_host) so it is reachable on the LAN;
keep the network in front of it trusted, or front it with a VPN/authenticating
proxy, and restrict the firewall to your subnet. Override with `web_host` or
`ideafindr web --host`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ideafindr import pipeline
from ideafindr.config import settings
from ideafindr.report.render import momentum_tag
from ideafindr.state import (
    EDITABLE,
    SECRET,
    apply_overrides,
    load_overrides,
    mask,
    save_overrides,
    secret_is_set,
)
from ideafindr.store import db
from ideafindr.web import browsercookies, jobs
from ideafindr.web.cookies import parse_cookie_input

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))

app = FastAPI(title="ideafindr")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

STANCE_LABEL = {
    "pain_point": "Pain point", "desire": "Desire", "objection": "Objection",
    "praise": "Praise", "question": "Question", "neutral": "Neutral",
}


def _conn():
    return db.connect()


def _runs() -> list:
    con = _conn()
    try:
        out = []
        for r in db.list_runs(con):
            out.append({
                "id": r.id, "topic": r.topic, "docs": r.doc_count,
                "when": r.created_at.strftime("%Y-%m-%d"),
                "themes": len(db.load_themes(con, r.id)),
                "has_report": (settings.reports_dir / r.id / "report.html").exists(),
            })
        return out
    finally:
        con.close()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request, "home.html",
        {"runs": _runs(), "active": [j.snapshot() for j in jobs.registry.active()],
         "sources": pipeline.ALL_SOURCES},
    )


@app.post("/research")
def research(
    topic: str = Form(...),
    days: int = Form(180),
    sources: list[str] = Form(default=[]),
):
    topic = topic.strip()
    if not topic:
        raise HTTPException(400, "topic is required")
    srcs = [s for s in sources if s in pipeline.ALL_SOURCES] or ["reddit", "web", "bluesky"]
    job = jobs.start_research(topic, days, srcs)
    return RedirectResponse(f"/research/{job.id}", status_code=303)


@app.get("/research/{job_id}", response_class=HTMLResponse)
def research_status(request: Request, job_id: str):
    job = jobs.registry.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return templates.TemplateResponse(request, "job.html", {"job": job.snapshot()})


@app.get("/api/jobs/{job_id}")
def job_json(job_id: str):
    job = jobs.registry.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.snapshot()


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str):
    con = _conn()
    try:
        run = db.get_run(con, run_id)
        if not run:
            raise HTTPException(404, "no such run")
        themes = db.load_themes(con, run_id)
        demand = db.load_demand_clusters(con, run_id)
        docs = list(db.iter_documents(con, run_id))
    finally:
        con.close()

    from ideafindr.analyze.signals import corpus_stats
    stats = corpus_stats(docs)
    report_exists = (settings.reports_dir / run_id / "report.html").exists()

    theme_rows = []
    for t in themes:
        cls, _ = momentum_tag(t.momentum)
        theme_rows.append({
            "label": t.label, "volume": t.volume, "momentum": t.momentum,
            "cls": cls, "stance": STANCE_LABEL.get(t.stance, t.stance),
            "coherent": t.coherent,
            "quotes": t.quotes[:2],
            "keywords": t.keywords[:6],
        })

    return templates.TemplateResponse(request, "run.html", {
        "run": run, "stats": stats, "themes": theme_rows, "demand": demand,
        "report_exists": report_exists,
    })


@app.post("/runs/{run_id}/analyze")
def run_analyze(run_id: str):
    job = jobs.start_action(run_id, _topic(run_id), "analyze")
    return RedirectResponse(f"/research/{job.id}", status_code=303)


@app.post("/runs/{run_id}/demand")
def run_demand(run_id: str):
    job = jobs.start_action(run_id, _topic(run_id), "demand")
    return RedirectResponse(f"/research/{job.id}", status_code=303)


@app.post("/runs/{run_id}/delete")
def run_delete(run_id: str):
    pipeline.delete_run(run_id)
    return RedirectResponse("/", status_code=303)


@app.get("/runs/{run_id}/report")
def run_report(run_id: str):
    path = settings.reports_dir / run_id / "report.html"
    if not path.exists():
        raise HTTPException(404, "no report yet; run the report stage first")
    return FileResponse(path, media_type="text/html")


def _topic(run_id: str) -> str:
    con = _conn()
    try:
        run = db.get_run(con, run_id)
        return run.topic if run else run_id
    finally:
        con.close()


# --- settings -------------------------------------------------------------------


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, saved: int = 0):
    return templates.TemplateResponse(request, "settings.html", {
        "s": settings, "saved": bool(saved),
        "secret_flags": {k: secret_is_set(load_overrides(), k) for k in SECRET},
        "cookie_available": browsercookies.available(),
    })


@app.post("/settings")
async def settings_save(request: Request):
    """Persist the editable settings. Unknown fields are dropped by save_overrides,
    and secret fields left blank are kept rather than wiped -- the form only ever
    shows a masked placeholder, so submitting it must not clear the stored key."""
    form = await request.form()
    updates: dict[str, object] = {}
    for key in EDITABLE:
        if key not in form:
            continue
        raw = str(form.get(key, "")).strip()
        if key in SECRET:
            # Blank means "leave as-is": the browser never received the value.
            if raw:
                updates[key] = raw
        elif key.startswith("enable_"):
            updates[key] = raw.lower() in ("1", "true", "on", "yes")
        else:
            updates[key] = raw
    save_overrides(updates)
    apply_overrides(settings)
    return RedirectResponse("/settings?saved=1", status_code=303)


# --- credentials ----------------------------------------------------------------


@app.get("/credentials", response_class=HTMLResponse)
def credentials_page(request: Request, saved: int = 0, error: str = ""):
    overrides = load_overrides()
    return templates.TemplateResponse(request, "credentials.html", {
        "s": settings,
        "saved": bool(saved), "error": error,
        "cookie_available": browsercookies.available(),
        "x_set": secret_is_set(overrides, "x_auth_token") or secret_is_set(overrides, "x_ct0"),
        "x_ct0_set": secret_is_set(overrides, "x_ct0"),
        "ig_set": secret_is_set(overrides, "instagram_sessionid"),
        "reddit_set": secret_is_set(overrides, "reddit_client_secret"),
        "x_masked": mask(str(overrides.get("x_auth_token", ""))),
        "ig_masked": mask(str(overrides.get("instagram_sessionid", ""))),
    })


@app.post("/credentials/x")
def credentials_x(auth_token: str = Form(""), ct0: str = Form("")):
    save_overrides({"x_auth_token": auth_token.strip(), "x_ct0": ct0.strip()})
    apply_overrides(settings)
    return RedirectResponse("/credentials?saved=1", status_code=303)


@app.post("/credentials/x/paste")
def credentials_x_paste(cookies: str = Form("")):
    """One-paste path: accept the whole browser `Cookie:` header.

    On a LAN the browser is on another machine, so `import from browser` (which
    reads the server's browser) cannot help, and `auth_token` is HttpOnly so the
    console cannot show it. The Cookie request header is the single copy that
    contains everything.
    """
    got = parse_cookie_input(cookies)
    token = got.get("auth_token", "")
    if not token:
        return RedirectResponse(
            "/credentials?error=No+auth_token+found.+Paste+the+whole+Cookie+"
            "header+from+DevTools+%E2%86%92+Network+%E2%86%92+Request+Headers.",
            status_code=303,
        )
    save_overrides({"x_auth_token": token, "x_ct0": got.get("ct0", "")})
    apply_overrides(settings)
    return RedirectResponse("/credentials?saved=1", status_code=303)


@app.post("/credentials/instagram")
def credentials_instagram(sessionid: str = Form(""), csrftoken: str = Form("")):
    save_overrides({
        "instagram_sessionid": sessionid.strip(),
        "instagram_csrftoken": csrftoken.strip(),
    })
    apply_overrides(settings)
    return RedirectResponse("/credentials?saved=1", status_code=303)


@app.post("/credentials/instagram/paste")
def credentials_instagram_paste(cookies: str = Form("")):
    got = parse_cookie_input(cookies)
    session = got.get("sessionid", "")
    if not session:
        return RedirectResponse(
            "/credentials?error=No+sessionid+found.+Paste+the+whole+Cookie+"
            "header+from+DevTools+%E2%86%92+Network+%E2%86%92+Request+Headers.",
            status_code=303,
        )
    save_overrides({
        "instagram_sessionid": session,
        "instagram_csrftoken": got.get("csrftoken", ""),
    })
    apply_overrides(settings)
    return RedirectResponse("/credentials?saved=1", status_code=303)


@app.post("/credentials/reddit")
def credentials_reddit(client_id: str = Form(""), client_secret: str = Form("")):
    save_overrides({
        "reddit_client_id": client_id.strip(),
        "reddit_client_secret": client_secret.strip(),
    })
    apply_overrides(settings)
    return RedirectResponse("/credentials?saved=1", status_code=303)


@app.post("/credentials/import/{platform}")
def credentials_import(platform: str):
    """Read cookies from a browser on this machine and store them.

    Same-machine only. This is the safest capture path: the platform saw a normal
    browser login, never automation.
    """
    if platform not in ("x", "instagram"):
        raise HTTPException(400, "unsupported platform")
    found = browsercookies.read_cookies(platform)
    if not found:
        return RedirectResponse(
            f"/credentials?error=No+{platform}+cookies+found+in+a+local+browser"
            "+profile.+Log+in+locally,+then+try+again+or+paste+manually.",
            status_code=303,
        )

    if platform == "x":
        if not found.get("auth_token"):
            return RedirectResponse(
                "/credentials?error=Found+an+x.com+session+without+auth_token",
                status_code=303,
            )
        save_overrides({
            "x_auth_token": found.get("auth_token", ""),
            "x_ct0": found.get("ct0", ""),
        })
    else:
        if not found.get("sessionid"):
            return RedirectResponse(
                "/credentials?error=Found+an+instagram.com+session+without+sessionid",
                status_code=303,
            )
        save_overrides({
            "instagram_sessionid": found.get("sessionid", ""),
            "instagram_csrftoken": found.get("csrftoken", ""),
        })
    apply_overrides(settings)
    return RedirectResponse("/credentials?saved=1", status_code=303)


@app.get("/health")
def health():
    con = _conn()
    try:
        return {"ok": True, "latest_run": db.latest_run_id(con)}
    finally:
        con.close()


def serve(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    settings.ensure_dirs()
    uvicorn.run(
        app,
        host=host or settings.web_host,
        port=port or settings.web_port,
        log_level="info",
    )
