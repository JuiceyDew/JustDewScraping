"""Open web: DuckDuckGo search + trafilatura extraction.

Free and key-less. Covers the reviews, forum threads and comparison posts that
live outside Reddit. Runs blocking libraries in a thread so it composes with the
async collectors.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from hashlib import sha1
from urllib.parse import urlparse

from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

# Aggregators and walled gardens: the extractor gets a login wall or a link farm,
# not an opinion.
_SKIP_DOMAINS = {
    "pinterest.com", "facebook.com", "instagram.com", "x.com", "twitter.com",
    "linkedin.com", "amazon.com", "ebay.com", "tiktok.com",
}
MIN_ARTICLE_CHARS = 400


def _domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.").lower()


def _search(query: str, n: int) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as d:
        return list(d.text(query, max_results=n))


def _extract(url: str) -> str | None:
    import trafilatura

    dl = trafilatura.fetch_url(url)
    if not dl:
        return None
    return trafilatura.extract(dl, include_comments=True, include_tables=False)


class WebCollector:
    """Note: `created_at` is collection time, not publication time.

    Trafilatura can return a publication date but it is missing or wrong often
    enough that trusting it would corrupt the trend charts. Web documents
    therefore contribute to themes and quotes but not to time-series signals.
    """

    name = "web"

    def __init__(self, per_query: int = 8):
        self.per_query = per_query

    async def collect(self, plan: RunPlan, limit: int = 40) -> list[Document]:
        queries = plan.web_queries or [plan.topic]
        seen: set[str] = set()
        hits: list[dict] = []

        for q in queries:
            try:
                results = await asyncio.to_thread(_search, q, self.per_query)
            except Exception as e:  # noqa: BLE001
                log.warning("web search failed for %r: %s", q, e)
                continue
            for r in results:
                url = r.get("href") or r.get("url") or ""
                dom = _domain(url)
                if not url or url in seen or any(dom.endswith(s) for s in _SKIP_DOMAINS):
                    continue
                seen.add(url)
                hits.append({"url": url, "title": r.get("title") or "", "domain": dom})
            if len(hits) >= limit:
                break

        hits = hits[:limit]
        texts = await asyncio.gather(
            *(asyncio.to_thread(_extract, h["url"]) for h in hits), return_exceptions=True
        )

        docs: list[Document] = []
        now = datetime.now(timezone.utc)
        for h, text in zip(hits, texts):
            if isinstance(text, BaseException) or not text:
                continue
            if len(text) < MIN_ARTICLE_CHARS:
                continue
            docs.append(
                Document(
                    id=f"web:{sha1(h['url'].encode()).hexdigest()[:16]}",
                    platform="web",
                    kind="article",
                    url=h["url"],
                    title=h["title"],
                    text=text[:20_000],
                    author=None,
                    created_at=now,
                    community=h["domain"],
                    engagement={},
                    raw={"domain": h["domain"]},
                )
            )
        return docs
