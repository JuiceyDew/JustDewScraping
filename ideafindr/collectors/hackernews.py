"""Hacker News via the Algolia search API.

Free, no key, no auth, and officially provided for HN rather than scraped, so
there is none of the ToS exposure the TikTok and Instagram collectors carry.

Skews technical and startup-shaped, which makes it strong for B2B, developer and
tooling topics and thin for consumer ones. That skew belongs in the report's
methodology caveats alongside the Reddit one.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from ideafindr.collectors.base import AsyncRateLimiter, is_on_topic, topic_terms
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

API = "https://hn.algolia.com/api/v1/search"
ITEM = "https://news.ycombinator.com/item?id="
MIN_COMMENT_CHARS = 80


def _doc(hit: dict, kind: str, story_score: int = 0) -> Document | None:
    oid = hit.get("objectID")
    if not oid:
        return None
    created = hit.get("created_at_i")
    if not created:
        return None

    if kind == "story":
        text = hit.get("story_text") or ""
        title = hit.get("title") or ""
        if not (title or text):
            return None
        score = int(hit.get("points") or 0)
        parent = None
    else:
        text = hit.get("comment_text") or ""
        if len(text) < MIN_COMMENT_CHARS:
            return None
        title = None
        # Comment hits carry no `points` field at all (verified against the live
        # API). Scoring them 0 would sink every HN comment to the bottom of quote
        # selection, so they inherit the score of the story they sit under --
        # crude, but it keeps them comparable to Reddit comments.
        score = story_score
        parent = f"hn:{hit.get('story_id')}" if hit.get("story_id") else None

    return Document(
        id=f"hn:{oid}",
        platform="hackernews",
        kind="post" if kind == "story" else "comment",
        url=f"{ITEM}{oid}",
        title=title,
        text=text,
        author=hit.get("author"),
        created_at=datetime.fromtimestamp(int(created), tz=timezone.utc),
        parent_id=parent,
        community="news.ycombinator.com",
        engagement={"score": score, "num_comments": int(hit.get("num_comments") or 0)},
        raw=hit,
    )


class HackerNewsCollector:
    name = "hackernews"

    def __init__(self, per_query: int = 50, rps: float = 5.0):
        self.per_query = per_query
        self._limiter = AsyncRateLimiter(rps)

    async def _search(
        self, client: httpx.AsyncClient, query: str, tags: str, after: int
    ) -> list[dict]:
        await self._limiter.wait()
        try:
            r = await client.get(
                API,
                params={
                    "query": query,
                    "tags": tags,
                    "hitsPerPage": self.per_query,
                    "numericFilters": f"created_at_i>{after}",
                },
            )
            if r.status_code != 200:
                log.warning("hn %r (%s): HTTP %s", query, tags, r.status_code)
                return []
            return r.json().get("hits", [])
        except Exception as e:  # noqa: BLE001
            log.warning("hn %r failed: %s", query, e)
            return []

    async def collect(self, plan: RunPlan, limit: int = 300) -> list[Document]:
        after = int((datetime.now(timezone.utc) - timedelta(days=plan.days)).timestamp())
        queries = [plan.topic, *plan.keywords[:5]]
        docs: dict[str, Document] = {}
        # Stories first, so their scores are known before comments are normalised.
        story_scores: dict[str, int] = {}
        terms = topic_terms(plan)
        dropped = 0
        # Half the budget for stories. Without a reserve, a broad topic fills the
        # whole limit with stories and the comments -- where the opinions are --
        # never get fetched at all.
        story_budget = max(1, limit // 2)

        async with httpx.AsyncClient(timeout=20) as client:
            for q in queries:
                for hit in await self._search(client, q, "story", after):
                    d = _doc(hit, "story")
                    if not d:
                        continue
                    # HN search is full-text over the whole site, so "cold plunge"
                    # returns "Stocks Could Plunge 70%". Gate on the distinctive
                    # topic words the way the Reddit collectors do.
                    if not is_on_topic(d, terms, min_terms=2):
                        dropped += 1
                        continue
                    docs.setdefault(d.id, d)
                    story_scores[str(hit.get("objectID"))] = int(hit.get("points") or 0)
                if len(docs) >= story_budget:
                    break

            for q in queries:
                for hit in await self._search(client, q, "comment", after):
                    d = _doc(hit, "comment",
                             story_score=story_scores.get(str(hit.get("story_id")), 0))
                    if not d:
                        continue
                    if not is_on_topic(d, terms, min_terms=2):
                        dropped += 1
                        continue
                    docs.setdefault(d.id, d)
                if len(docs) >= limit:
                    break

        out = list(docs.values())[:limit]
        log.info("hackernews: %d documents (%d stories), %d off-topic dropped",
                 len(out), sum(1 for d in out if d.kind == "post"), dropped)
        return out
