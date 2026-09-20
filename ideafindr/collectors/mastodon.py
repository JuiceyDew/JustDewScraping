"""Mastodon collector - federated social network.

Mastodon is a free, federated social network (like Twitter but decentralized).
The API is open and generous - no authentication needed for public posts.

Free tier: Unlimited public reads (rate limited per instance)
ToS: Explicitly allows API access to public posts
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ideafindr.collectors.base import Collector
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

_TAGS = re.compile(r"<[^>]+>")
_ENTITIES = re.compile(r"&[a-zA-Z]+;|&#\d+;")

# Popular instances to search
INSTANCES = [
    "https://mastodon.social",
    "https://fosstodon.org",  # Tech-focused
    "https://mastodon.online",
    "https://mastodon.world",
]

MAX_LIMIT = 40


class MastodonCollector(Collector):
    """Collect public posts from Mastodon instances."""

    name = "mastodon"

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _ensure_client(self) -> httpx.Client:
        if not self._client:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={"User-Agent": "ideafindr/0.1 (market research)"},
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    def _search_instance(
        self, instance: str, query: str, limit: int = MAX_LIMIT
    ) -> list[dict[str, Any]]:
        """Search posts on a specific Mastodon instance."""
        client = self._ensure_client()
        params = {"q": query, "limit": limit, "resolve": "true"}
        
        try:
            r = client.get(
                f"{instance}/api/v2/search",
                params=params,
            )
            if r.status_code == 429:
                log.warning("mastodon: rate limited on %s", instance)
                return []
            r.raise_for_status()
            data = r.json()
            statuses = data.get("statuses", [])
            log.info("mastodon: %s returned %d posts for %r", instance, len(statuses), query)
            return statuses
        except httpx.HTTPError as e:
            log.warning("mastodon: error on %s: %s", instance, e)
            return []

    def _parse_post(self, post: dict[str, Any], instance: str) -> Document | None:
        """Parse a Mastodon post (toot) into our Document schema."""
        try:
            # Strip HTML tags from content and collapse entities/whitespace.
            content = post.get("content", "")
            text = _TAGS.sub(" ", content)
            text = _ENTITIES.sub(" ", text)
            text = " ".join(text.split()).strip()

            if not text:
                return None
            
            # Engagement
            favorites = post.get("favourites_count", 0)
            reposts = post.get("reblogs_count", 0)
            replies = post.get("replies_count", 0)
            
            # Timestamp
            created_at_str = post.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now(timezone.utc)
            
            # Author
            account = post.get("account", {})
            author = account.get("acct", "")
            author_display = account.get("display_name", "")
            author_str = f"{author_display} (@{author})" if author_display else f"@{author}"
            
            # Build IDs. `instance` is a full base URL; fall back to the whole
            # string if it is not, rather than crashing on split('//')[1].
            post_id = post.get("id", "")
            host = instance.split("//", 1)[1] if "//" in instance else instance
            doc_id = f"mastodon:{host}:{post_id}"
            
            # URL
            url = post.get("url", f"{instance}/@{author}/{post_id}")
            
            return Document(
                id=doc_id,
                platform="mastodon",
                kind="post",
                url=url,
                title=None,
                text=text[:2000],  # Mastodon posts can be long
                author=author_str,
                created_at=created_at,
                parent_id=None,
                community=instance,
                engagement={
                    "likes": favorites,
                    "reposts": reposts,
                    "replies": replies,
                    "score": favorites + 2 * reposts + 3 * replies,
                },
                raw=post,
            )
        except Exception as e:
            log.warning("mastodon: failed to parse post: %s", e)
            return None

    async def collect(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        """Collect Mastodon posts matching the plan."""
        docs: dict[str, Document] = {}
        
        # Build queries
        queries = []
        if plan.topic:
            queries.append(plan.topic)
        for kw in plan.keywords[:5]:
            queries.append(kw)
        for brand in plan.brands[:3]:
            queries.append(brand)
        
        log.info("mastodon: searching %d queries across %d instances", len(queries), len(INSTANCES))
        
        per_query = max(10, limit // (len(queries) * len(INSTANCES)))
        
        for query in queries:
            for instance in INSTANCES:
                try:
                    posts = self._search_instance(instance, query, per_query)
                    
                    for post in posts:
                        doc = self._parse_post(post, instance)
                        if doc:
                            # Check date window
                            days_old = (datetime.now(timezone.utc) - doc.created_at).days
                            if days_old <= plan.days:
                                docs.setdefault(doc.id, doc)
                    
                    if len(docs) >= limit:
                        log.info("mastodon: reached limit of %d documents", limit)
                        break
                        
                except Exception as e:
                    log.error("mastodon: error on %s for %r: %s", instance, query, e)
            
            if len(docs) >= limit:
                break
        
        result = list(docs.values())
        log.info("mastodon: collected %d posts", len(result))
        return result

    def collect_sync(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        """Synchronous wrapper."""
        import asyncio
        return asyncio.run(self.collect(plan, limit))
