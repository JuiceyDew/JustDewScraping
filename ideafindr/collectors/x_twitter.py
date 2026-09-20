"""X/Twitter via twscrape. Cookie-based, no API key.

twscrape drives X's internal GraphQL endpoints using a browser session. It is the
only viable route: X's paid API is out of scope, and the unauthenticated
syndication endpoints no longer return search results.

Auth is a browser cookie pair, not a login. `auth_token` identifies the account
and `ct0` is the CSRF token every GraphQL call needs. Both are entered through
the web UI and stored in the state dir (never in the repo).

    Use a dedicated burner account. Never a client's or a personal one.

Low profile by design, because this deploys without proxies:
  * one account, no rotation;
  * a conservative per-collection cap;
  * twscrape's own rate-limit accounting, which stops an account two requests
    before X's limit rather than charging through it;
  * telemetry disabled.

Failure is contained: a missing cookie, a suspended account or a changed query
id degrades this collector to zero documents. It never fails the run (the
pipeline wraps every collector in `safe_collect`).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

from ideafindr.collectors.base import is_on_topic, topic_terms
from ideafindr.config import settings
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

# twscrape keeps its account pool and session cookies in its own SQLite file.
ACCOUNT_DB = "x_accounts.db"
# The local identifier for the single account. twscrape does not verify it
# against the cookie's real username, so a fixed name is fine.
ACCOUNT_NAME = "ideafindr"

# X's search API is expensive to hammer and this is a single account from one
# residential IP. Keep the default cap low; a run that wants more can pass it.
DEFAULT_PER_QUERY = 40
DEFAULT_QUERIES = 6


def _account_db_path():
    from ideafindr.state import resolve_state_dir

    return resolve_state_dir() / ACCOUNT_DB


def has_credentials() -> bool:
    return bool(settings.x_auth_token and settings.x_ct0)


async def _ensure_account(api) -> bool:
    """Register the cookie pair with twscrape's pool, replacing any prior session.

    Re-adding on every run is intentional: cookies expire, and twscrape's
    `add_account_cookies` preserves the account's stats/locks while refreshing the
    session.
    """
    cookies = f"auth_token={settings.x_auth_token}; ct0={settings.x_ct0}"
    try:
        await api.pool.add_account_cookies(ACCOUNT_NAME, cookies)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("x: could not register the cookie account: %s", e)
        return False


def _tweet_to_doc(tweet) -> Document | None:
    """Map a twscrape Tweet onto our Document schema.

    A retweet or pure reply-to-other is real content but its `rawContent` is the
    original's; keeping it would double-count the same words. Quotes are kept --
    the comment text is the quoter's own.
    """
    tid = getattr(tweet, "id_str", None) or str(getattr(tweet, "id", "") or "")
    if not tid:
        return None
    text = (getattr(tweet, "rawContent", "") or "").strip()
    if not text:
        return None

    user = getattr(tweet, "user", None)
    username = getattr(user, "username", None) if user else None
    display = getattr(user, "displayname", None) if user else None
    author = f"{display} (@{username})" if display and username else (
        f"@{username}" if username else None
    )

    created = getattr(tweet, "date", None)
    if not isinstance(created, datetime):
        created = datetime.now(timezone.utc)

    url = getattr(tweet, "url", None) or (
        f"https://x.com/{username}/status/{tid}" if username else f"https://x.com/i/status/{tid}"
    )

    # Hashtags are the closest thing X has to a community label.
    tags = getattr(tweet, "hashtags", None) or []
    community = f"#{tags[0]}" if tags else None

    views = getattr(tweet, "viewCount", None) or 0
    return Document(
        id=f"x:{tid}",
        platform="x",
        kind="post",
        url=url,
        title=None,
        text=text,
        author=author,
        created_at=created,
        parent_id=None,
        community=community,
        engagement={
            "likes": getattr(tweet, "likeCount", 0) or 0,
            "reposts": getattr(tweet, "retweetCount", 0) or 0,
            "num_comments": getattr(tweet, "replyCount", 0) or 0,
            "quotes": getattr(tweet, "quoteCount", 0) or 0,
            "views": views if isinstance(views, int) else 0,
            "score": (
                (getattr(tweet, "likeCount", 0) or 0)
                + 2 * (getattr(tweet, "retweetCount", 0) or 0)
            ),
        },
        raw={"lang": getattr(tweet, "lang", None)},
    )


def _queries(plan: RunPlan, max_queries: int) -> list[str]:
    """Search terms, ordered so the most specific survive the cap."""
    out: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)

    if plan.topic:
        add(plan.topic)
    for kw in plan.keywords:
        add(kw)
    for brand in plan.brands[:3]:
        add(brand)
    # Exclude retweets -- they dominate search and are not anyone's own words.
    return [f"{q} -filter:retweets" for q in out[:max_queries]]


async def _collect_async(plan: RunPlan, limit: int) -> list[Document]:
    # Imported lazily so the package stays optional-ish and import cost is only
    # paid when X is actually requested.
    from twscrape import API, gather

    if not has_credentials():
        return []

    db = _account_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)

    api = API(
        str(db),
        raise_when_no_account=True,
        wait_timeout=20,
        wait_interval=2,
    )
    if not await _ensure_account(api):
        return []

    terms = topic_terms(plan)
    per_query = max(10, min(DEFAULT_PER_QUERY, limit // DEFAULT_QUERIES))
    docs: dict[str, Document] = {}

    for q in _queries(plan, DEFAULT_QUERIES):
        if len(docs) >= limit:
            break
        try:
            found = await gather(api.search(q, limit=per_query))
        except Exception as e:  # noqa: BLE001 - rate limits, lockouts, query-id churn
            log.warning("x: search %r failed: %s", q, e)
            continue

        for tweet in found:
            d = _tweet_to_doc(tweet)
            if not d:
                continue
            # X search is a whole-site search, so apply the same two-term gate
            # the other site-wide collectors use; one generic word in common is
            # coincidence ("plunge" -> stock market posts).
            if not is_on_topic(d, terms, min_terms=2):
                continue
            docs.setdefault(d.id, d)

    log.info("x: collected %d posts", len(docs))
    return list(docs.values())


class XCollector:
    name = "x"

    def __init__(self, limit_cap: int = 120):
        self.limit_cap = limit_cap

    async def collect(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        if not has_credentials():
            log.warning(
                "x: skipping -- no auth_token/ct0. Add them under Credentials in "
                "the web UI (use a dedicated account, not a personal one)."
            )
            return []
        # Twscrape telemetry sends operation names to the maintainers. Off.
        os.environ.setdefault("TWS_TELEMETRY", "0")
        cap = min(limit, self.limit_cap)
        try:
            return await _collect_async(plan, cap)
        except Exception as e:  # noqa: BLE001
            log.warning("x collector failed: %s", e)
            return []

    def collect_sync(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        return asyncio.run(self.collect(plan, limit))
