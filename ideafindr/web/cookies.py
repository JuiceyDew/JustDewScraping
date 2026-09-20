"""Parse the many shapes a user's pasted cookies can take.

The credentials page asks for one paste instead of one field per cookie name,
because on a LAN the browser is on a different machine than the server: the
"import from browser" path cannot see it, and the cookies that actually matter
(`auth_token`, `sessionid`) are HttpOnly, so `document.cookie` in the console
omits them. The one reliable, one-shot copy is the browser's request `Cookie:`
header, which includes HttpOnly values.

This accepts, in any mix:

  * a Cookie request header / `document.cookie` string:
        auth_token=abc; ct0=xyz
  * newline- or semicolon-separated name=value pairs
  * a JSON object: {"auth_token": "abc", "ct0": "xyz"}
  * a Netscape cookies.txt export (tab-separated, name at col 5, value at 6)

It never raises: unparseable input yields {}, and the route reports what was
missing. Cookie values are opaque -- a trailing '=' in a base64 token is kept.
"""

from __future__ import annotations

import json
import re

# Name is everything up to the first '=' or whitespace; the value is the rest of
# the line, so '=' inside a value survives. Quotes are stripped below.
_PAIR = re.compile(r"^\s*([^=\s]+)\s*=\s*(.*?)\s*$")


def parse_cookie_input(raw: str) -> dict[str, str]:
    """Extract {name: value} from whatever the user pasted."""
    raw = (raw or "").strip()
    if not raw:
        return {}

    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, dict):
            return {
                str(k).strip(): str(v).strip()
                for k, v in data.items()
                if str(v).strip()
            }

    out: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Netscape cookies.txt: domain, flag, path, secure, expiry, name, value.
        if "\t" in line:
            cols = line.split("\t")
            if len(cols) >= 7 and cols[5].strip():
                out[cols[5].strip()] = cols[6].strip()
                continue
        for part in line.split(";"):
            m = _PAIR.match(part.strip())
            if not m:
                continue
            name = m.group(1)
            value = m.group(2).strip().strip('"')
            if value:
                out[name] = value
    return out
