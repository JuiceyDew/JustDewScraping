"""Runtime state and secrets, kept out of the repository.

The web UI lets a user enter the Ollama key, model choices and social session
cookies without editing `.env`. Those values are written here -- to a
`settings.json` in the state directory -- and never into the repo. The state
directory defaults to `data/`, which is already gitignored, and can be relocated
with `IDEAFINDR_STATE_DIR` (the NixOS module points it at the systemd
StateDirectory).

Precedence, lowest to highest:

    defaults  <  .env  <  settings.json  <  real environment

so an operator can always override whatever the UI stored via the environment
(or a Nix `EnvironmentFile`), and the UI cannot silently shadow it.

The file is written atomically and chmod 0600 because it holds live session
cookies and an API key.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Keys the UI is allowed to read/write. Anything not listed here is ignored, so
# a malicious or buggy form post cannot rewrite `db_path` to an arbitrary path.
EDITABLE: frozenset[str] = frozenset({
    # Ollama Cloud
    "ollama_api_key",
    "ollama_base_url",
    "fast_model",
    "smart_model",
    # embeddings
    "embed_backend",
    "embed_model",
    "embed_base_url",
    "embed_api_key",
    # Reddit
    "reddit_client_id",
    "reddit_client_secret",
    "reddit_user_agent",
    # X / Twitter
    "x_auth_token",
    "x_ct0",
    # Instagram
    "instagram_session_user",
    "instagram_sessionid",
    "instagram_csrftoken",
    # optional scrapers
    "enable_tiktok",
    "enable_instagram",
    "tiktok_ms_token",
    # web UI bind (takes effect on restart)
    "web_host",
    "web_port",
    # web UI password (empty disables the login gate)
    "auth_password",
})

# The subset that must never be rendered back to a page or written to a log.
SECRET: frozenset[str] = frozenset({
    "ollama_api_key",
    "embed_api_key",
    "reddit_client_secret",
    "x_auth_token",
    "x_ct0",
    "instagram_sessionid",
    "instagram_csrftoken",
    "tiktok_ms_token",
    "auth_password",
})


def resolve_state_dir() -> Path:
    """Where the database, reports and settings live.

    Resolution order:

    1. ``IDEAFINDR_STATE_DIR`` -- set explicitly, or by the NixOS module to the
       systemd ``StateDirectory``.
    2. ``<repo>/data`` when the package is running from a writable source
       checkout (the normal development case).
    3. ``$XDG_STATE_HOME/ideafindr`` (``~/.local/state/ideafindr``) otherwise.

    Step 3 matters for an installed package: under Nix, ``ROOT`` is a read-only
    store path, so ``ROOT/data`` cannot be created and the app would crash on
    first run without a writable default.
    """
    env = os.environ.get("IDEAFINDR_STATE_DIR", "").strip()
    if env:
        return Path(env).expanduser()

    from ideafindr.config import ROOT

    if os.access(ROOT, os.W_OK):
        return ROOT / "data"

    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
    return base / "ideafindr"


def settings_file() -> Path:
    return resolve_state_dir() / "settings.json"


def load_overrides() -> dict[str, Any]:
    """Read settings.json. A missing or corrupt file yields {} rather than raising:
    a bad edit should degrade to defaults, not brick the app."""
    path = settings_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("ignoring unreadable %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in EDITABLE}


def save_overrides(values: dict[str, Any]) -> None:
    """Merge `values` into settings.json. Only EDITABLE keys are kept.

    Written atomically with mode 0600: a partial file would lose every stored
    session, and a world-readable one would leak the API key.
    """
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_overrides()
    for k, v in values.items():
        if k not in EDITABLE:
            continue
        # An empty string means "clear it", which is legitimate for a key the
        # user is rotating; store the empty value rather than dropping the key.
        current[k] = v
    _atomic_write(path, json.dumps(current, indent=2))
    try:
        path.chmod(0o600)
    except OSError as e:  # pragma: no cover - non-POSIX filesystems
        log.warning("could not chmod %s: %s", path, e)


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".settings-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def mask(value: str) -> str:
    """A display form for a secret that proves something is stored without
    revealing it. Never return the raw value to the browser."""
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}{'*' * 8}{value[-2:]}"


def secret_is_set(values: dict[str, Any], key: str) -> bool:
    return bool(str(values.get(key) or "").strip())


def apply_overrides(settings: Any) -> None:
    """Push stored overrides onto a live Settings instance.

    Used after a save to pick up a changed key without a restart. Keys present in
    the real environment are skipped, matching the construction-time precedence
    (environment > settings.json): a NixOS `EnvironmentFile` must not be
    silently overridden by whatever the UI stored.
    """
    overrides = load_overrides()
    for k, v in overrides.items():
        if k not in EDITABLE or not hasattr(settings, k):
            continue
        if os.environ.get(k.upper()):
            continue
        try:
            setattr(settings, k, v)
        except Exception as e:  # noqa: BLE001
            log.warning("could not apply override %s: %s", k, e)
