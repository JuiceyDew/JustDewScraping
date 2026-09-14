"""Lemmy: federated, Reddit-shaped, free, no auth.

The data model is Reddit's -- communities, posts with a title and body, threaded
comments, scores -- so it normalises almost directly onto Document. Volume is far
lower than Reddit's, so treat it as a supplement that keeps one struggling service
from being the entire corpus, not as a replacement.

Federation means the same post is visible from many instances under different
local ids. `ap_id` is its ActivityPub identity and is stable across all of them,
so that is what dedupes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx

from ideafindr.collectors.base import AsyncRateLimiter, is_on_topic, topic_terms
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

# Large, general instances. Each federates a different slice, so querying a few
# and deduping on ap_id gives better coverage than hammering one.
INSTANCES = ["lemmy.world", "lemmy.ml", "sh.itjust.works"]
MIN_COMMENT_CHARS = 60


def _when(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Lemmy serves ISO 8601, sometimes with a Z and sometimes naive.
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def post_to_doc(view: dict) -> Document | None:
    post = view.get("post") or {}
    ap_id = post.get("ap_id")
    created = _when(post.get("published"))
    if not ap_id or not created:
        return None
    if post.get("deleted") or post.get("removed"):
        return None
    counts = view.get("counts") or {}
    community = (view.get("community") or {}).get("name")
    return Document(
        # ap_id is the cross-instance identity: the same post seen from three
        # instances has three local ids but one ap_id.
        id=f"lemmy:{ap_id}",
        platform="lemmy",
        kind="post",
        url=ap_id,
        title=post.get("name"),
        text=post.get("body") or "",
        author=(view.get("creator") or {}).get("name"),
        created_at=created,
        community=community,
        engagement={
            "score": int(counts.get("score") or 0),
            "num_comments": int(counts.get("comments") or 0),
        },
        raw={"instance": view.get("_instance"), "ap_id": ap_id},
    )


def comment_to_doc(view: dict, parent_ap_id: str) -> Document | None:
    comment = view.get("comment") or {}
    ap_id = comment.get("ap_id")
    created = _when(comment.get("published"))
    body = (comment.get("content") or "").strip()
    if not ap_id or not created or len(body) < MIN_COMMENT_CHARS:
        return None
    if comment.get("deleted") or comment.get("removed"):
        return None
    counts = view.get("counts") or {}
    return Document(
        id=f"lemmy:{ap_id}",
        platform="lemmy",
        kind="comment",
        url=ap_id,
        title=None,
        text=body,
        author=(view.get("creator") or {}).get("name"),
        created_at=created,
        parent_id=f"lemmy:{parent_ap_id}",
        community=(view.get("community") or {}).get("name"),
        engagement={"score": int(counts.get("score") or 0)},
        raw={"ap_id": ap_id},
    )


class LemmyCollector:
    name = "lemmy"

    def __init__(self, instances: list[str] | None = None, rps: float = 2.0,
                 with_comments: bool = True):
        self.instances = instances or list(INSTANCES)
        self._limiter = AsyncRateLimiter(rps)
        self.with_comments = with_comments

    async def _get(self, client: httpx.AsyncClient, instance: str, path: str,
                   params: dict) -> dict | None:
        await self._limiter.wait()
        try:
            r = await client.get(f"https://{instance}/api/v3{path}", params=params)
            if r.status_code != 200:
                log.debug("lemmy %s%s: HTTP %s", instance, path, r.status_code)
                return None
            return r.json()
        except Exception as e:  # noqa: BLE001
            log.debug("lemmy %s%s failed: %s", instance, path, e)
            return None

    async def _comments(self, client: httpx.AsyncClient, instance: str,
                        post_id: int, ap_id: str) -> list[Document]:
        body = await self._get(client, instance, "/comment/list",
                               {"post_id": post_id, "limit": 30, "sort": "Top",
                                "max_depth": 2})
        if not body:
            return []
        out = []
        for v in body.get("comments", []):
            d = comment_to_doc(v, ap_id)
            if d:
                out.append(d)
        return out

    async def collect(self, plan: RunPlan, limit: int = 200) -> list[Document]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=plan.days)
        queries = [plan.topic, *plan.keywords[:4]]
        docs: dict[str, Document] = {}
        terms = topic_terms(plan)
        dropped = 0
        # (instance, local post id, ap_id) for the posts worth pulling comments on
        threads: list[tuple[str, int, str]] = []

        async with httpx.AsyncClient(
            timeout=20, headers={"User-Agent": "ideafindr/0.1 (research)"}
        ) as client:
            for instance in self.instances:
                for q in queries:
                    if len(docs) >= limit:
                        break
                    body = await self._get(client, instance, "/search",
                                           {"q": q, "type_": "Posts",
                                            "sort": "TopAll", "limit": 25})
                    if not body:
                        continue
                    for view in body.get("posts", []):
                        view["_instance"] = instance
                        d = post_to_doc(view)
                        if not d or d.created_at < cutoff or d.id in docs:
                            continue
                        # Lemmy search is site-wide, so a generic keyword returns
                        # "How do you guys make your coffee?".
                        if not is_on_topic(d, terms, min_terms=2):
                            dropped += 1
                            continue
                        docs[d.id] = d
                        pid = (view.get("post") or {}).get("id")
                        ap = (view.get("post") or {}).get("ap_id")
                        if pid and ap and d.engagement.get("num_comments", 0) > 1:
                            threads.append((instance, int(pid), ap))

            if self.with_comments:
                for instance, pid, ap in threads[:40]:
                    for d in await self._comments(client, instance, pid, ap):
                        docs.setdefault(d.id, d)

        out = list(docs.values())[:limit]
        log.info("lemmy: %d documents across %d instance(s), %d off-topic dropped",
                 len(out), len(self.instances), dropped)
        return out
