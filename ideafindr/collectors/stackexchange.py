"""Stack Exchange: 170+ Q&A sites, free, no key required.

High signal where it applies -- answers are voted, edited and dated -- and
irrelevant where it does not, which is most consumer topics. The planner therefore
chooses the sites (`RunPlan.stack_sites`) and this collector does nothing at all
when it chose none. That is the correct outcome for "cold plunge tubs"; it is the
wrong outcome to spend the daily quota discovering it.

QUOTA. 300 requests/day/IP unkeyed, 10,000 with a free key from stackapps.com.
That is small enough that an unbudgeted collector would exhaust it on one run and
then fail silently for the rest of the day, so requests are hard-capped per run and
the remaining quota the API reports is logged.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timedelta, timezone

import httpx

from ideafindr.collectors.base import AsyncRateLimiter, is_on_topic, topic_terms
from ideafindr.config import settings
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

API = "https://api.stackexchange.com/2.3"
# `withbody` is required; without it the API returns titles and scores but no text,
# which would give the analysis nothing to cluster.
FILTER = "withbody"
# Unkeyed callers get 300/day. One run must not be able to spend it all.
MAX_REQUESTS_PER_RUN = 12

_TAG = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    """Bodies come back as HTML. Code blocks are kept as text -- in a Q&A about a
    product, the "code" is often a spec or a parts list."""
    return " ".join(html.unescape(_TAG.sub(" ", s or "")).split())


def _doc(item: dict, site: str) -> Document | None:
    qid = item.get("question_id") or item.get("answer_id")
    created = item.get("creation_date")
    if not qid or not created:
        return None
    body = _strip_html(item.get("body") or "")
    title = html.unescape(item.get("title") or "") or None
    if not body and not title:
        return None
    return Document(
        id=f"se:{site}:{qid}",
        platform="stackexchange",
        kind="post",
        url=item.get("link") or f"https://{site}.stackexchange.com/q/{qid}",
        title=title,
        text=body,
        author=((item.get("owner") or {}).get("display_name")),
        created_at=datetime.fromtimestamp(int(created), tz=timezone.utc),
        community=site,
        engagement={
            "score": int(item.get("score") or 0),
            "num_comments": int(item.get("answer_count") or 0),
        },
        raw={"site": site, "tags": item.get("tags") or []},
    )


class StackExchangeCollector:
    name = "stackexchange"

    def __init__(self, sites: list[str] | None = None, rps: float = 2.0):
        self.sites = sites
        self._limiter = AsyncRateLimiter(rps)
        self._spent = 0

    async def _search(self, client: httpx.AsyncClient, site: str, q: str,
                      after: int) -> list[dict]:
        if self._spent >= MAX_REQUESTS_PER_RUN:
            return []
        self._spent += 1
        await self._limiter.wait()
        params = {
            "order": "desc", "sort": "votes", "q": q, "site": site,
            "filter": FILTER, "pagesize": 30, "fromdate": after,
        }
        if settings.stackexchange_key:
            params["key"] = settings.stackexchange_key
        try:
            r = await client.get(f"{API}/search/advanced", params=params)
            if r.status_code != 200:
                log.warning("stackexchange %s %r: HTTP %s", site, q, r.status_code)
                return []
            body = r.json()
            remaining = body.get("quota_remaining")
            if remaining is not None and remaining < 30:
                log.warning("stackexchange quota nearly spent: %s left today", remaining)
            if body.get("backoff"):
                log.warning("stackexchange asked for a %ss backoff", body["backoff"])
            return body.get("items", [])
        except Exception as e:  # noqa: BLE001
            log.warning("stackexchange %s %r failed: %s", site, q, e)
            return []

    async def collect(self, plan: RunPlan, limit: int = 150) -> list[Document]:
        sites = self.sites if self.sites is not None else list(plan.stack_sites)
        if not sites:
            log.info("stackexchange: no sites chosen for this topic; skipping")
            return []

        after = int((datetime.now(timezone.utc) - timedelta(days=plan.days)).timestamp())
        queries = [plan.topic, *plan.keywords[:3]]
        docs: dict[str, Document] = {}
        terms = topic_terms(plan)

        async with httpx.AsyncClient(timeout=20) as client:
            for site in sites[:3]:
                for q in queries:
                    if len(docs) >= limit or self._spent >= MAX_REQUESTS_PER_RUN:
                        break
                    for item in await self._search(client, site, q, after):
                        d = _doc(item, site)
                        if d and is_on_topic(d, terms, min_terms=2):
                            docs.setdefault(d.id, d)

        out = list(docs.values())[:limit]
        log.info("stackexchange: %d documents from %s (%d requests)",
                 len(out), ", ".join(sites[:3]), self._spent)
        return out
