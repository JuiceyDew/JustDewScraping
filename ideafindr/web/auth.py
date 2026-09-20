"""A single-password login gate for the web UI.

The server is headless and the UI holds an API key and live session cookies, so
binding it to 0.0.0.0 without a gate means anyone on the LAN can read the
Settings/Credentials pages or delete a project. This is a small self-contained
login: one password, a signed session cookie, no external proxy, no user table.

The password comes from, in precedence order:

  * ``AUTH_PASSWORD`` in the real environment (what the NixOS ``environmentFile``
    or systemd ``Environment=`` sets), or
  * ``auth_password`` in ``settings.json``, set with ``ideafindr passwd`` or on
    the Settings page once logged in.

When it is empty the gate is disabled and the UI behaves exactly as before, so
development and the offline test suite are unaffected.

The cookie is signed with HMAC-SHA256 using a random secret persisted in the
state directory (mode 0600), so sessions survive a restart. Session tokens are
stdlib-only -- no new dependency.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from pathlib import Path

from ideafindr.config import settings

log = logging.getLogger(__name__)

COOKIE = "ideafindr_session"
# 30 days. A homelab tool should not log you out every day.
SESSION_TTL = 30 * 24 * 3600


def enabled() -> bool:
    return bool((settings.auth_password or "").strip())


def check_password(candidate: str) -> bool:
    """Constant-time comparison, so a wrong guess leaks no timing signal."""
    expected = (settings.auth_password or "").strip()
    if not expected:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def _secret() -> bytes:
    """A stable signing key, persisted so sessions outlive a restart.

    Falls back to a per-process key if the state directory is not writable, which
    only means sessions reset on restart -- better than refusing to start.
    """
    from ideafindr.state import resolve_state_dir

    path = resolve_state_dir() / ".web-secret"
    try:
        if path.exists():
            raw = path.read_bytes().strip()
            if raw:
                return raw
        key = secrets.token_bytes(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write 0600 from creation: the key signs every session cookie.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
        return key
    except OSError as e:  # pragma: no cover - read-only state dir
        log.warning("could not persist the web session secret: %s", e)
        return secrets.token_bytes(32)


def _sign(payload: bytes) -> str:
    return hmac.new(_secret(), payload, hashlib.sha256).hexdigest()


def make_token() -> str:
    payload = str(int(time.time())).encode("ascii")
    # Strip base64 padding: '=' is not a valid cookie-octet, so a padded value
    # gets quoted by clients (and then fails verification when sent back).
    b64 = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"{b64}.{_sign(payload)}"


def valid_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    b64, _, sig = token.partition(".")
    # Restore the padding stripped in make_token.
    b64 += "=" * (-len(b64) % 4)
    try:
        payload = base64.urlsafe_b64decode(b64.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return False
    if not hmac.compare_digest(sig, _sign(payload)):
        return False
    try:
        issued = int(payload.decode("ascii"))
    except ValueError:
        return False
    return (time.time() - issued) < SESSION_TTL


def secret_path() -> Path:
    from ideafindr.state import resolve_state_dir

    return resolve_state_dir() / ".web-secret"
