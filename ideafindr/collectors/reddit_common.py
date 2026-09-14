"""Reddit normalisation and filtering, shared by every Reddit source.

Arctic Shift, Reddit's own API and any future mirror all serve the *same* field
names -- Arctic Shift is a Reddit archive, so its records are Reddit's records.
Only the transport and the envelope differ (Reddit wraps each item in
`{"kind": "t3", "data": {...}}`; Arctic Shift returns the inner object directly).

Everything that interprets those fields therefore lives here rather than in one
collector, so a fix to bot detection or the topic gate applies to all of them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ideafindr.collectors.base import TOPIC_STOP, is_on_topic, topic_terms, topic_words
from ideafindr.models import Document

# Re-exported: these are source-agnostic and live in base.py, but Reddit
# collectors and their tests have always imported them from here.
__all__ = [
    "TOO_BIG", "SWEEP_MAX_SUBSCRIBERS", "TOPIC_STOP", "is_on_topic", "topic_terms",
    "topic_words", "post_to_doc", "comment_to_doc", "is_bot", "unwrap", "ts",
    "permalink",
]

# Keyword search is unsupported on very high-traffic subreddits; skip them and
# rely on smaller, topic-native communities where the signal is better anyway.
TOO_BIG = 3_000_000
# Above this, a keyword-less sweep returns generic front-page content, not topic
# discussion. Determined by watching r/Biohackers (~1M) flood a cold-plunge corpus
# with 534 off-topic posts about boron, peptides and green powders.
SWEEP_MAX_SUBSCRIBERS = 250_000

def ts(v: Any) -> datetime:
    return datetime.fromtimestamp(float(v or 0), tz=timezone.utc)


def permalink(raw: dict) -> str:
    pl = raw.get("permalink") or ""
    if pl.startswith("/"):
        return f"https://www.reddit.com{pl}"
    return pl or raw.get("url") or ""


def post_to_doc(raw: dict) -> Document | None:
    pid = raw.get("id")
    if not pid:
        return None
    return Document(
        id=f"reddit:{pid}",
        platform="reddit",
        kind="post",
        url=permalink(raw),
        title=raw.get("title"),
        text=raw.get("selftext") or "",
        author=raw.get("author"),
        created_at=ts(raw.get("created_utc")),
        parent_id=None,
        community=raw.get("subreddit"),
        engagement={
            "score": raw.get("score") or 0,
            "num_comments": raw.get("num_comments") or 0,
            "upvote_ratio": raw.get("upvote_ratio"),
        },
        raw=raw,
    )


# Moderator bots post identical boilerplate under thousands of threads. Left in,
# they form their own large "theme" and poison the language bank.
BOT_AUTHORS = {"automoderator", "[deleted]", "amputatorbot", "remindmebot", "sneakpeekbot"}
BOT_PHRASES = ("i am a bot", "this action was performed automatically", "beep boop")


def is_bot(author: str | None, text: str) -> bool:
    a = (author or "").lower()
    if a in BOT_AUTHORS or a.endswith("bot") or a.endswith("-bot"):
        return True
    low = text.lower()
    return any(p in low for p in BOT_PHRASES)


def comment_to_doc(raw: dict) -> Document | None:
    cid = raw.get("id")
    if not cid:
        return None
    link_id = (raw.get("link_id") or "").removeprefix("t3_")
    body = raw.get("body") or ""
    if body in ("[deleted]", "[removed]", ""):
        return None
    if is_bot(raw.get("author"), body):
        return None
    return Document(
        id=f"reddit:{cid}",
        platform="reddit",
        kind="comment",
        url=permalink(raw),
        title=None,
        text=body,
        author=raw.get("author"),
        created_at=ts(raw.get("created_utc")),
        parent_id=f"reddit:{link_id}" if link_id else None,
        community=raw.get("subreddit"),
        engagement={"score": raw.get("score") or 0},
        raw=raw,
    )


def unwrap(item: dict) -> dict:
    """Reddit's API wraps every item as {"kind": "t3", "data": {...}}.

    Arctic Shift serves the inner object directly. Accepting both here is what
    lets one set of normalisers serve both sources.
    """
    if isinstance(item, dict) and "data" in item and "kind" in item:
        return item["data"] or {}
    return item
