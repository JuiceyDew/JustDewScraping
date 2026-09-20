"""Read social-session cookies from a browser on this machine.

Same-machine only. The web UI runs on the machine whose browser you logged into,
so it can read that browser's cookie store directly -- the safest way to obtain a
session, because the platform only ever saw a normal, un-automated login.

On Linux, Firefox cookies decrypt without any external keyring; Chromium-family
browsers need the OS keyring, which is why a `browser_cookie3` read can return
nothing in a headless/systemd context. Every function here returns {} on any
failure rather than raising, so the UI can fall back to manual paste.

This never logs or returns cookie values except to the caller, which writes them
straight to the 0600 state file.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# (domain, cookie-name) pairs we want per platform.
TARGETS = {
    "x": [("x.com", "auth_token"), ("x.com", "ct0"), ("twitter.com", "auth_token"),
          ("twitter.com", "ct0")],
    "instagram": [("instagram.com", "sessionid"), ("instagram.com", "csrftoken")],
    "reddit": [("reddit.com", "token_v2"), ("reddit.com", "reddit_session")],
}


def available() -> bool:
    try:
        import browser_cookie3  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _browsers():
    """Browser readers in the order most likely to succeed on a Linux desktop."""
    import browser_cookie3

    return [
        ("firefox", browser_cookie3.firefox),
        ("chrome", browser_cookie3.chrome),
        ("chromium", browser_cookie3.chromium),
        ("brave", browser_cookie3.brave),
        ("edge", browser_cookie3.edge),
    ]


def read_cookies(platform: str) -> dict[str, str]:
    """Return {cookie_name: value} for a platform, or {} if none were found.

    Tries each installed browser; the first that yields the platform's key
    cookies wins.
    """
    wanted = TARGETS.get(platform)
    if not wanted or not available():
        return {}

    names = {n for _, n in wanted}
    domains = {d for d, _ in wanted}

    for label, reader in _browsers():
        try:
            jar = reader(domain_name="")
        except Exception as e:  # noqa: BLE001 - missing profile, locked keyring, etc.
            log.debug("cookie read via %s failed: %s", label, e)
            continue

        out: dict[str, str] = {}
        for cookie in jar:
            domain = (cookie.domain or "").lstrip(".")
            if domain in domains and cookie.name in names and cookie.value:
                out[cookie.name] = cookie.value
        # Require the identifying cookie, not just any.
        if any(n in out for n in names):
            log.info("credentials: read %s cookies from %s", platform, label)
            return out
    return {}


def read_x() -> dict[str, str]:
    return read_cookies("x")


def read_instagram() -> dict[str, str]:
    return read_cookies("instagram")
