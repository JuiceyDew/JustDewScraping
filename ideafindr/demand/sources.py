"""Autocomplete endpoints -- the demand-side equivalent of a collector.

These are the endpoints a browser's search bar calls. They need no key, no auth
and no circumvention, which is why they are usable here at all. They are also
**undocumented and unsupported**: nobody promises this response shape, and any of
them can change or start throttling without notice. That is a materially weaker
guarantee than Arctic Shift's, so every parser here is defensive and every source
is allowed to fail independently -- a dead endpoint should thin the query space,
never fail the run.

Verified live 2026-09-13: all four answer HTTP 200 with no key, and -- usefully --
all four put the suggestion list at index 1 of a JSON array, so one parser covers
them:

    google   [query, [suggestions], [], {metadata}]
    youtube  same shape, different suggestions (ds=yt selects YouTube's index)
    ddg      [query, [suggestions]]          (type=list -> OpenSearch)
    bing     [query, [suggestions], ...]     (osjson -> OpenSearch)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Autocomplete is personalised and regionalised. Pinning language and region is
# what makes two runs of the same topic comparable; without it the query space
# drifts with whatever the endpoint infers about the caller.
HL, GL = "en", "us"

# A bare httpx user-agent gets throttled noticeably sooner on these endpoints.
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

SOURCES: dict[str, dict[str, Any]] = {
    "google": {
        "url": "https://suggestqueries.google.com/complete/search",
        "params": {"client": "firefox", "hl": HL, "gl": GL},
        "qkey": "q",
    },
    "youtube": {
        "url": "https://suggestqueries.google.com/complete/search",
        "params": {"client": "firefox", "ds": "yt", "hl": HL, "gl": GL},
        "qkey": "q",
    },
    "ddg": {
        "url": "https://duckduckgo.com/ac/",
        "params": {"type": "list", "kl": f"{GL}-{HL}"},
        "qkey": "q",
    },
    "bing": {
        "url": "https://api.bing.com/osjson.aspx",
        "params": {},
        "qkey": "query",
    },
}

ALL_SOURCES = list(SOURCES)


def parse_suggestions(payload: Any) -> list[str]:
    """Pull the suggestion list out of an OpenSearch-shaped response.

    Tolerant on purpose: these endpoints are unversioned. Anything that is not a
    list of strings at index 1 yields nothing rather than raising, so one provider
    changing its format degrades the query space instead of killing the harvest.
    """
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    raw = payload[1]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        # DuckDuckGo has historically returned [{"phrase": ...}] without
        # type=list; accept both rather than depending on the parameter.
        if isinstance(item, dict):
            item = item.get("phrase") or item.get("text") or ""
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


async def fetch_suggestions(
    client: httpx.AsyncClient, source: str, term: str
) -> list[str]:
    """One (source, term) lookup. Returns [] on any failure."""
    cfg = SOURCES[source]
    params = {**cfg["params"], cfg["qkey"]: term}
    try:
        r = await client.get(cfg["url"], params=params, headers={"User-Agent": UA})
    except Exception as e:  # noqa: BLE001 - a dead endpoint must not fail a run
        log.debug("suggest %s %r failed: %s", source, term, e)
        return []
    if r.status_code != 200:
        log.debug("suggest %s %r -> HTTP %s", source, term, r.status_code)
        return []
    try:
        # Google serves this as text/javascript, so r.json() is not guaranteed to
        # be offered by content type -- parse the body directly.
        import json

        return parse_suggestions(json.loads(r.text))
    except Exception as e:  # noqa: BLE001
        log.debug("suggest %s %r unparseable: %s", source, term, e)
        return []
