"""Reddit's own API. The fix for the rate limiting.

Arctic Shift is one person's free archive and it shows: measured 2026-09-14, it
answered `422 "Timeout. Maybe slow down a bit"` to 3 of 14 keyword queries at the
2 req/s we already throttle ourselves to. A collection fires 40-60 such queries,
so roughly a fifth of every corpus was being lost -- and each failure triggers the
halving-retry ladder, putting *more* load on the service that just asked us to
slow down.

Reddit's own Data API allows 100 queries/minute with OAuth, and its keyword search
is a first-class index rather than a scan, so it does not time out. That is ~45x
the throughput at a fraction of the failure rate.

    LICENSING. The free tier is for NON-COMMERCIAL use -- personal projects,
    research, bots, moderator tools. Selling reports built on this data needs
    Reddit's approval (manual, weeks) and is billed per call. This is a licensing
    constraint, not a technical one, so nothing here will warn you when you cross
    it. See the README.

Credentials: reddit.com/prefs/apps -> "create app" -> type "script" -> free. Only
the client id and secret are needed; read-only application-only OAuth wants no
Reddit password.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

from ideafindr.collectors.base import AsyncRateLimiter
from ideafindr.collectors.reddit_common import (
    SWEEP_MAX_SUBSCRIBERS,
    TOO_BIG,
    comment_to_doc,
    is_on_topic,
    post_to_doc,
    topic_terms,
    topic_words,
    unwrap,
)
from ideafindr.config import settings
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API = "https://oauth.reddit.com"

# 90/min against Reddit's 100/min ceiling. The headroom matters because the limit
# is averaged over a rolling window and a retry can otherwise tip us over it.
DEFAULT_RPS = 1.5


class RedditAuthError(RuntimeError):
    """Bad or missing credentials. Not retryable -- it will fail identically."""


class RedditAPIClient:
    """Application-only OAuth against Reddit's Data API."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        rps: float | None = None,
    ):
        self.client_id = client_id if client_id is not None else settings.reddit_client_id
        self.client_secret = (
            client_secret if client_secret is not None else settings.reddit_client_secret
        )
        # Reddit rejects generic user-agents outright, and asks that it identify
        # the app and its author.
        self.user_agent = user_agent or settings.reddit_user_agent
        self._limiter = AsyncRateLimiter(rps or DEFAULT_RPS)
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0), headers={"User-Agent": self.user_agent}
        )

    async def __aenter__(self) -> "RedditAPIClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def token(self) -> str:
        """Fetch or reuse the bearer token.

        grant_type=client_credentials is application-only OAuth: it authenticates
        the app, not a user, which is all read-only access needs.
        """
        async with self._lock:
            if self._token and time.time() < self._expires_at - 60:
                return self._token
            if not self.client_id or not self.client_secret:
                raise RedditAuthError(
                    "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set. Create a "
                    "free app at https://reddit.com/prefs/apps (type: script) and "
                    "put the id and secret in .env. Without them the pipeline falls "
                    "back to Arctic Shift, which is rate limited."
                )
            r = await self._client.post(
                TOKEN_URL,
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
            )
            if r.status_code in (401, 403):
                raise RedditAuthError(
                    f"Reddit rejected the credentials (HTTP {r.status_code}). Check "
                    "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET, and that the app "
                    "type is 'script'."
                )
            r.raise_for_status()
            body = r.json()
            self._token = body["access_token"]
            self._expires_at = time.time() + float(body.get("expires_in", 3600))
            log.info("reddit api: token acquired, valid ~%.0f min",
                     float(body.get("expires_in", 3600)) / 60)
            return self._token

    async def get(self, path: str, params: dict | None = None) -> dict | list | None:
        """One rate-limited request. Returns None on a non-fatal failure."""
        tok = await self.token()
        await self._limiter.wait()
        try:
            r = await self._client.get(
                f"{API}{path}",
                params={**(params or {}), "raw_json": 1},
                headers={"Authorization": f"Bearer {tok}"},
            )
        except Exception as e:  # noqa: BLE001
            log.warning("reddit api %s failed: %s", path, e)
            return None

        if r.status_code == 429:
            # Reddit reports the window in headers; wait it out once rather than
            # hammering. Exceeding the limit repeatedly is how apps get banned.
            reset = float(r.headers.get("x-ratelimit-reset", 10) or 10)
            log.warning("reddit api: rate limited, sleeping %.0fs", reset)
            await asyncio.sleep(min(reset, 60))
            return None
        if r.status_code in (401, 403):
            # The token may simply have expired early; drop it and let the caller
            # retry, rather than failing the whole run.
            self._token = None
            log.warning("reddit api %s: HTTP %s", path, r.status_code)
            return None
        if r.status_code >= 400:
            log.warning("reddit api %s: HTTP %s %s", path, r.status_code, r.text[:120])
            return None
        try:
            return r.json()
        except Exception:  # noqa: BLE001
            return None

    async def listing(self, path: str, params: dict | None = None) -> list[dict]:
        """A Reddit listing, unwrapped to the inner records."""
        body = await self.get(path, params)
        if not isinstance(body, dict):
            return []
        children = body.get("data", {}).get("children", [])
        return [unwrap(c) for c in children if isinstance(c, dict)]


class RedditAPICollector:
    name = "reddit"

    def __init__(self, client: RedditAPIClient | None = None):
        self._client = client

    async def about(self, cl: RedditAPIClient, sub: str) -> tuple[str, int] | None:
        """Resolve a subreddit name to its real casing and subscriber count."""
        body = await cl.get(f"/r/{sub}/about")
        if not isinstance(body, dict):
            return None
        data = body.get("data") or {}
        name = data.get("display_name")
        if not name:
            return None
        return name, int(data.get("subscribers") or 0)

    async def resolve_subreddits(
        self, cl: RedditAPIClient, plan: RunPlan
    ) -> list[tuple[str, int]]:
        """Validate the planner's guesses. It invents plausible communities."""
        out: list[tuple[str, int]] = []
        seen: set[str] = set()
        for raw in plan.subreddits:
            n = raw.removeprefix("r/").removeprefix("/r/").strip()
            if not n or n.lower() in seen:
                continue
            got = await self.about(cl, n)
            if not got:
                continue
            name, subs = got
            if subs > TOO_BIG:
                log.info("skipping r/%s: %s subscribers, too big to keyword-search",
                         name, f"{subs:,}")
                continue
            seen.add(name.lower())
            out.append((name, subs))
        return out

    async def discover_subreddits(
        self, cl: RedditAPIClient, plan: RunPlan, want: int = 6
    ) -> list[tuple[str, int]]:
        """Ask Reddit what communities exist for the topic.

        Scored the same way as the Arctic path: a candidate needs two topic terms
        in its name or description, because one generic word in common is
        coincidence (searching "cold" returns r/Coldplay).
        """
        words = topic_words(plan.topic)
        if not words:
            return []
        terms = set(words) | {w.rstrip("s") for w in words}
        scored: dict[str, tuple[int, int]] = {}

        for q in [" ".join(words), *words[:3]]:
            rows = await cl.listing("/subreddits/search", {"q": q, "limit": 25})
            for d in rows:
                name = d.get("display_name")
                subs = int(d.get("subscribers") or 0)
                if not name or not (200 <= subs <= SWEEP_MAX_SUBSCRIBERS):
                    continue
                hay = f"{name} {d.get('public_description') or ''} {d.get('title') or ''}".lower()
                hay_words = set(hay.split())
                score = len(terms & (hay_words | {w.rstrip("s") for w in hay_words}))
                if score < 2:
                    continue
                prev = scored.get(name)
                if not prev or score > prev[0]:
                    scored[name] = (score, subs)

        top = sorted(scored.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[:want]
        if top:
            log.info("discovered %s", ", ".join(f"r/{n}({v[1]:,})" for n, v in top))
        return [(n, v[1]) for n, v in top]

    async def search_posts(
        self, cl: RedditAPIClient, sub: str, keyword: str | None, limit: int
    ) -> list[Document]:
        """One (subreddit, keyword) slice.

        No timeout-recovery ladder here, unlike the Arctic path: Reddit's search is
        a real index rather than a scan over an archive, so it either answers or
        returns an error worth surfacing.
        """
        if keyword:
            rows = await cl.listing(
                f"/r/{sub}/search",
                {"q": keyword, "restrict_sr": 1, "sort": "relevance",
                 "t": "all", "limit": min(limit, 100)},
            )
        else:
            rows = await cl.listing(f"/r/{sub}/new", {"limit": min(limit, 100)})
        return [d for d in (post_to_doc(r) for r in rows) if d]

    async def fetch_comments(
        self, cl: RedditAPIClient, post_id: str, limit: int = 60
    ) -> list[Document]:
        """The comments are where the opinions are; titles are just topics."""
        native = post_id.removeprefix("reddit:")
        body = await cl.get(f"/comments/{native}", {"limit": limit, "depth": 2,
                                                    "sort": "top"})
        if not isinstance(body, list) or len(body) < 2:
            return []
        children = body[1].get("data", {}).get("children", [])
        out: list[Document] = []
        for c in children:
            d = unwrap(c) if isinstance(c, dict) else {}
            if not d or d.get("body") is None:
                continue  # "more" placeholders carry no body
            doc = comment_to_doc(d)
            if doc:
                out.append(doc)
        return out

    async def collect(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=plan.days)
        own = self._client is None
        cl = self._client or RedditAPIClient()
        try:
            subs = await self.resolve_subreddits(cl, plan)
            if len(subs) < 3:
                log.info("only %d subreddit(s) resolved; discovering more", len(subs))
                known = {n.lower() for n, _ in subs}
                subs += [d for d in await self.discover_subreddits(cl, plan)
                         if d[0].lower() not in known]
            if not subs:
                log.warning("no subreddits found for %r; nothing to collect", plan.topic)
                return []
            log.info("collecting from %d subreddit(s) via reddit api: %s",
                     len(subs), ", ".join(f"r/{n}({s:,})" for n, s in subs))

            keywords = list(plan.keywords[:6])
            per_slice = max(25, limit // max(1, len(subs) * (len(keywords) + 1)))
            posts: dict[str, Document] = {}
            terms = topic_terms(plan)
            dropped = stale = 0

            for sub, subscribers in subs:
                slices: list[str | None] = list(keywords)
                if subscribers <= SWEEP_MAX_SUBSCRIBERS:
                    slices.append(None)  # keyword-less sweep, small subs only
                gate = subscribers > SWEEP_MAX_SUBSCRIBERS
                for kw in slices:
                    found = await self.search_posts(cl, sub, kw, per_slice)
                    for d in found:
                        # Reddit's search has no date filter worth trusting, so the
                        # window is applied here instead.
                        if d.created_at < cutoff:
                            stale += 1
                            continue
                        if gate and not is_on_topic(d, terms):
                            dropped += 1
                            continue
                        posts.setdefault(d.id, d)

            if dropped or stale:
                log.info("dropped %d off-topic and %d out-of-window posts", dropped, stale)
            docs: list[Document] = list(posts.values())

            ranked = sorted(posts.values(),
                            key=lambda d: d.engagement.get("num_comments", 0), reverse=True)
            targets = [d for d in ranked if d.engagement.get("num_comments", 0) > 1][:60]
            for d in targets:
                docs += await self.fetch_comments(cl, d.id)

            log.info("reddit api: %d posts, %d documents total", len(posts), len(docs))
            return docs
        finally:
            if own:
                await cl.__aexit__()
