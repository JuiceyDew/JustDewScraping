"""TikTok via the unofficial TikTok-Api wrapper. OPTIONAL, OFF BY DEFAULT.

This scrapes TikTok without using an official API, which is against TikTok's
Terms of Service. For a marketing company selling research to clients that is a
commercial and legal exposure, not just a technical one -- see the README.

It is also unreliable by nature: TikTok changes its internals frequently and
detects datacenter IPs, so expect `EmptyResponseException` without residential
proxies. Every failure here degrades the corpus but must never fail the run
(see collectors/base.safe_collect).

Requires:  uv sync --extra scrapers  &&  uv run playwright install chromium
Plus an `ms_token` cookie from a logged-out tiktok.com session in TIKTOK_MS_TOKEN.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ideafindr.config import settings
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)


def _video_to_doc(v: dict, hashtag: str | None = None) -> Document | None:
    vid = v.get("id")
    if not vid:
        return None
    desc = (v.get("desc") or "").strip()
    if not desc:
        return None
    author = (v.get("author") or {})
    handle = author.get("uniqueId") if isinstance(author, dict) else str(author)
    stats = v.get("stats") or {}
    ts = v.get("createTime")
    return Document(
        id=f"tiktok:{vid}",
        platform="tiktok",
        kind="video",
        url=f"https://www.tiktok.com/@{handle}/video/{vid}" if handle else f"https://www.tiktok.com/video/{vid}",
        title=None,
        text=desc,
        author=handle,
        created_at=datetime.fromtimestamp(int(ts), tz=timezone.utc) if ts else datetime.now(timezone.utc),
        community=f"#{hashtag}" if hashtag else None,
        engagement={
            "likes": stats.get("diggCount", 0),
            "views": stats.get("playCount", 0),
            "num_comments": stats.get("commentCount", 0),
            "shares": stats.get("shareCount", 0),
        },
        raw=v,
    )


class TikTokCollector:
    name = "tiktok"

    def __init__(self, per_tag: int = 30, with_comments: bool = True):
        self.per_tag = per_tag
        self.with_comments = with_comments

    async def collect(self, plan: RunPlan, limit: int = 200) -> list[Document]:
        from TikTokApi import TikTokApi

        token = settings.tiktok_ms_token
        if not token:
            log.warning("TIKTOK_MS_TOKEN is empty; TikTok will almost certainly return nothing")

        docs: list[Document] = []
        tags = plan.hashtags or [plan.topic.replace(" ", "")]

        async with TikTokApi() as api:
            await api.create_sessions(
                ms_tokens=[token] if token else None,
                num_sessions=1,
                sleep_after=3,
                browser="chromium",
            )
            for tag in tags:
                if len(docs) >= limit:
                    break
                try:
                    async for video in api.hashtag(name=tag).videos(count=self.per_tag):
                        d = _video_to_doc(video.as_dict, hashtag=tag)
                        if not d:
                            continue
                        docs.append(d)

                        if self.with_comments and d.engagement.get("num_comments", 0) > 5:
                            try:
                                async for c in video.comments(count=20):
                                    cd = c.as_dict
                                    text = (cd.get("text") or "").strip()
                                    if not text:
                                        continue
                                    docs.append(
                                        Document(
                                            id=f"tiktok:c{cd.get('cid')}",
                                            platform="tiktok",
                                            kind="comment",
                                            url=d.url,
                                            text=text,
                                            author=(cd.get("user") or {}).get("unique_id"),
                                            created_at=datetime.fromtimestamp(
                                                int(cd.get("create_time", 0)) or 0, tz=timezone.utc
                                            ),
                                            parent_id=d.id,
                                            community=f"#{tag}",
                                            engagement={"likes": cd.get("digg_count", 0)},
                                            raw=cd,
                                        )
                                    )
                            except Exception as e:  # noqa: BLE001
                                log.debug("tiktok comments failed for %s: %s", d.id, e)
                except Exception as e:  # noqa: BLE001
                    log.warning("tiktok hashtag #%s failed (%s) -- bot detection is the usual "
                                "cause; a residential proxy is normally required", tag, e)
        return docs[:limit]
