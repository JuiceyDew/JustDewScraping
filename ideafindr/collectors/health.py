"""Live credential checks for the cookie/credential-based collectors.

Session cookies expire, and a dead cookie does not fail a run -- `safe_collect`
swallows it and the collector returns zero documents. That is the right runtime
behaviour, but it means an expired X or Instagram session looks exactly like
"nobody is talking about this topic". `ideafindr doctor` and `ideafindr accounts`
run the checks here to tell the two apart before a run wastes ten minutes.

Every function is defensive: a missing optional dependency, an unconfigured
credential or a network error is reported as a status tuple, never raised.
"""

from __future__ import annotations

import logging

from ideafindr.config import settings

log = logging.getLogger(__name__)

# (ok, detail). `ok` is True only when a live check succeeded. An unconfigured
# source is (False, "..."), which callers render as a warning, not a failure --
# none of these sources is required for the pipeline to produce a report.
Status = tuple[bool, str]


def probe_x() -> Status:
    """Is the stored X cookie pair still a live session?

    This is the same wiring the collector uses (register the cookies with
    twscrape's pool, then hit the API once), so a green probe means a collection
    will not silently return nothing.
    """
    from ideafindr.collectors import x_twitter

    if not x_twitter.has_credentials():
        return False, "not configured (no auth_token/ct0)"
    try:
        from twscrape import API, gather
    except ImportError:
        return False, "twscrape not installed (uv sync)"

    import asyncio
    import os

    async def _run() -> Status:
        db = x_twitter._account_db_path()
        db.parent.mkdir(parents=True, exist_ok=True)
        api = API(str(db), raise_when_no_account=True, wait_timeout=15, wait_interval=2)
        if not await x_twitter._ensure_account(api):
            return False, "could not register the cookie pair"
        try:
            # One cheap call: if the session is dead, X answers 401/403 here.
            found = await gather(api.search("hello", limit=1))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "401" in msg or "403" in msg or "unauthorized" in msg.lower():
                return False, "session rejected (expired cookie?) -- re-paste"
            return False, f"request failed: {msg[:60]}"
        return True, f"live session ({len(found)} result probe)"

    os.environ.setdefault("TWS_TELEMETRY", "0")
    try:
        return asyncio.run(_run())
    except Exception as e:  # noqa: BLE001 - never let a probe crash doctor
        return False, f"error: {str(e)[:70]}"


def probe_instagram() -> Status:
    """Is there a usable Instagram session, and if so does it still log in?"""
    try:
        import instaloader  # noqa: F401
    except ImportError:
        return False, "instaloader not installed (uv sync --extra scrapers)"

    from ideafindr.collectors import instagram

    if not instagram.has_session():
        return False, "not configured (no sessionid or session file)"
    try:
        L = instagram._loader()
        username = L.test_login()
    except Exception as e:  # noqa: BLE001
        return False, f"error: {str(e)[:70]}"
    if username:
        return True, f"live session (user {username})"
    return False, "session rejected (expired?) -- re-paste or refresh the file"


def probe_reddit() -> Status:
    """Which Reddit transport is active, and is it reachable?"""
    if settings.reddit_client_id and settings.reddit_client_secret:
        return True, "official API configured (OAuth)"

    # No credentials: the fallback is Arctic Shift, which the doctor already
    # probes directly. Report the selection rather than duplicating that call.
    return False, "no API creds; using Arctic Shift (see arctic: probes above)"
