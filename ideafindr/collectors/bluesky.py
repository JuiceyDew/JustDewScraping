"""Bluesky collector via the AT Protocol.

Bluesky's AT Protocol is designed for open access: no API keys required for
reading public posts. The search endpoint returns posts matching a query,
with engagement metrics (likes, reposts, replies) and timestamps.

Free tier: Unlimited reads for public data (rate limited to ~100 req/min)
ToS: Explicitly allows scraping via API for public data
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ideafindr.collectors.base import Collector, is_on_topic, topic_terms
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

ATPROTO_HOST = "https://public.api.bsky.app"
MAX_LIMIT = 100


class BlueskyCollector(Collector):
    """Collect public posts from Bluesky via the AT Protocol."""

    name = "bluesky"

    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(
                base_url=ATPROTO_HOST,
                timeout=httpx.Timeout(60.0, connect=15.0, read=45.0),
                headers={"User-Agent": "ideafindr/0.1 (market research)"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    async def _search_posts(self, query: str, limit: int = MAX_LIMIT) -> list[dict[str, Any]]:
        """Search posts on Bluesky.

        The public API endpoint needs no authentication and returns posts
        newest-first. The call is async so it overlaps with the other collectors
        rather than blocking the event loop.
        """
        client = self._ensure_client()
        params = {
            "q": query,
            "limit": min(limit, MAX_LIMIT),
            "sort": "latest",
        }
        r = await client.get("/xrpc/app.bsky.feed.searchPosts", params=params)
        r.raise_for_status()
        posts = r.json().get("posts", [])
        log.info("bluesky: %d posts for %r", len(posts), query)
        return posts

    def _parse_post(self, post: dict[str, Any]) -> Document | None:
        """Parse a Bluesky post into our Document schema."""
        try:
            record = post.get("record", {})
            author = post.get("author", {})
            
            # Extract text - Bluesky posts can have facets (rich text)
            text = record.get("text", "")
            
            # Parse facets for mentions/hashtags if present
            hashtags: list[str] = []
            mentions: list[str] = []
            for facet in record.get("facets", []):
                features = facet.get("features", [])
                for feature in features:
                    if feature.get("$type") == "app.bsky.richtext.facet#tag":
                        hashtags.append(feature.get("tag", ""))
                    elif feature.get("$type") == "app.bsky.richtext.facet#mention":
                        mentions.append(feature.get("did", ""))
            
            # Engagement metrics
            likes = post.get("likeCount", 0)
            reposts = post.get("repostCount", 0)
            replies = post.get("replyCount", 0)
            
            # Parse timestamp
            created_at_str = record.get("createdAt", "")
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now(timezone.utc)
            
            # Build document ID
            post_uri = post.get("uri", "")  # at://did:plc:.../app.bsky.feed.post/...
            native_id = post_uri.split("/")[-1] if "/" in post_uri else post_uri
            doc_id = f"bluesky:{native_id}"
            
            # Author handle
            author_handle = author.get("handle", "")
            author_display = author.get("displayName", "")
            author_str = f"{author_display} (@{author_handle})" if author_display else f"@{author_handle}"
            
            # Community: use the author's handle as a proxy for community
            community = f"u/{author_handle}" if author_handle else None
            
            return Document(
                id=doc_id,
                platform="bluesky",
                kind="post",
                url=f"https://bsky.app/profile/{author_handle}/post/{native_id}",
                title=None,
                text=text,
                author=author_str,
                created_at=created_at,
                parent_id=None,
                community=community,
                engagement={
                    "likes": likes,
                    "reposts": reposts,
                    "replies": replies,
                    "score": likes + 2 * reposts + 3 * replies,
                },
                raw=post,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("bluesky: failed to parse post: %s", e)
            return None

    async def collect(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        """Collect Bluesky posts matching the plan.

        Bluesky's search is keyword-based, so the query set comes from the
        planner's keywords and brands. Bluesky has no community scoping, which
        makes it a site-wide search: the two-term on-topic gate is what keeps a
        generic keyword from dragging in unrelated posts.
        """
        docs: dict[str, Document] = {}
        queries_seen: set[str] = set()

        queries: list[str] = []
        for kw in plan.keywords:
            queries.append(kw)
        for brand in plan.brands:
            for kw in plan.keywords[:3]:
                queries.append(f"{brand} {kw}")
        if plan.topic:
            queries.append(plan.topic)

        for q in queries:
            if q.lower() not in queries_seen:
                queries_seen.add(q.lower())

        log.info("bluesky: searching %d queries", len(queries_seen))
        terms = topic_terms(plan)
        per_query_limit = max(20, min(60, limit // max(1, len(queries_seen))))

        try:
            for query in queries_seen:
                try:
                    posts = await self._search_posts(query, limit=per_query_limit)
                    for post in posts:
                        doc = self._parse_post(post)
                        if not doc or not doc.text.strip():
                            continue
                        days_old = (datetime.now(timezone.utc) - doc.created_at).days
                        if days_old > plan.days:
                            continue
                        if not is_on_topic(doc, terms, min_terms=2):
                            continue
                        docs.setdefault(doc.id, doc)

                    if len(docs) >= limit:
                        log.info("bluesky: reached limit of %d documents", limit)
                        break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        log.warning("bluesky: rate limited, stopping early")
                        break
                    log.error("bluesky: HTTP error for %r: %s", query, e)
                except Exception as e:  # noqa: BLE001
                    log.error("bluesky: error collecting for %r: %s", query, e)
        finally:
            await self.aclose()

        result = list(docs.values())
        log.info("bluesky: collected %d posts", len(result))
        return result

    def collect_sync(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        """Synchronous wrapper for scripts that are not already async."""
        return asyncio.run(self.collect(plan, limit))
