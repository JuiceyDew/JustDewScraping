"""Reddit via Arctic Shift (https://arctic-shift.photon-reddit.com).

Free, public, no auth. Two constraints discovered by probing the live API that
shape this whole collector:

1. `query`/`selftext`/`body` REQUIRE an `author` or `subreddit` scope. A naive
   "search all of Reddit for X" returns HTTP 400. So we resolve subreddits first,
   then fan out per-subreddit.
2. The service can return **HTTP 200 with `"data": null`** when it throttles or a
   query is too expensive. That is a retryable condition, not an empty result --
   treating it as empty silently loses data.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ideafindr.collectors.base import AsyncRateLimiter
from ideafindr.collectors.reddit_common import (
    SWEEP_MAX_SUBSCRIBERS,
    TOO_BIG,
    comment_to_doc,
    is_bot,
    is_on_topic,
    post_to_doc,
    topic_terms,
    topic_words,
)
from ideafindr.config import settings
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)


class ArcticTransient(Exception):
    """429, 5xx, or the data:null throttle response. Retry it."""


class ArcticClient:
    """Rate-limited async client for the Arctic Shift API."""

    def __init__(self, base: str | None = None, rps: float | None = None):
        self.base = (base or settings.arctic_base).rstrip("/")
        self._limiter = AsyncRateLimiter(rps or settings.arctic_rps)
        # Increased timeout: Arctic Shift can be slow on complex queries
        # 120s total, with 30s for connect, 60s for pool
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=30.0, pool=60.0),
            headers={"User-Agent": "ideafindr/0.1 (research)"},
        )

    async def __aenter__(self) -> "ArcticClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type(ArcticTransient),
        stop=stop_after_attempt(5),  # Increased from 4 to 5 for reliability
        wait=wait_exponential(multiplier=2, min=2, max=40),  # Increased max wait to 40s
        reraise=True,
    )
    async def _get_once(self, path: str, params: dict[str, Any]) -> list[dict] | None:
        """One request. Returns None when the backend timed out on the query
        itself (as opposed to being busy), which the caller answers by asking for
        a smaller page rather than by waiting."""
        await self._limiter.wait()
        try:
            r = await self._client.get(f"{self.base}{path}", params=params)
        except httpx.ConnectTimeout as e:
            log.warning("arctic: connect timeout on %s: %s", path, e)
            raise ArcticTransient(f"connect timeout on {path}")
        except httpx.ReadTimeout as e:
            log.warning("arctic: read timeout on %s: %s", path, e)
            raise ArcticTransient(f"read timeout on {path}")
        except httpx.HTTPError as e:
            log.error("arctic: HTTP error on %s: %s", path, e)
            return []

        if r.status_code == 429 or r.status_code >= 500:
            raise ArcticTransient(f"HTTP {r.status_code} on {path}")
        if r.status_code in (400, 422):
            if "timeout" in r.text.lower():
                return None
            # A real query error (e.g. unscoped keyword search, bad parameter).
            # Never retry -- it will fail identically.
            log.warning("arctic %d on %s %s: %s", r.status_code, path, params, r.text[:200])
            return []
        r.raise_for_status()

        body = r.json()
        data = body.get("data")
        if data is None:
            # HTTP 200 + data:null == throttled. Retryable, unlike a query timeout.
            raise ArcticTransient(f"data:null on {path} (throttled)")
        return data

    async def get(self, path: str, params: dict[str, Any]) -> list[dict]:
        """Fetch, halving the page size when the backend times out on the query.

        "Timeout. Maybe slow down a bit" means the query was too expensive to
        execute, not that we are sending too many requests -- so the fix is a
        smaller page, not a longer wait. Retrying the identical request (which is
        all tenacity can do) just fails four times and loses the slice.
        """
        clean = {k: v for k, v in params.items() if v is not None and v != ""}
        limit = int(clean.get("limit", 0))

        for attempt in range(5):  # Increased from 4 to 5
            data = await self._get_once(path, clean)
            if data is not None:
                return data
            if limit <= 10:
                if attempt < 4:  # Try one more time at minimum limit before giving up
                    log.info("arctic: retrying %s at minimum limit=%d (attempt %d/5)", path, limit, attempt + 2)
                    continue
                log.warning("arctic query timeout on %s even at limit=%d; giving up",
                            path, limit)
                return []
            limit = max(10, limit // 2)
            clean = dict(clean, limit=limit)
            log.info("arctic query timeout on %s; retrying with limit=%d", path, limit)
        return []


# --- collector ----------------------------------------------------------------


class RedditArcticCollector:
    name = "reddit"

    def __init__(self, client: ArcticClient | None = None):
        self._client = client

    async def discover_subreddits(
        self, cl: ArcticClient, plan: RunPlan, want: int = 6
    ) -> list[tuple[str, int]]:
        """Find real subreddits for the topic via the API.

        The planner *guesses* community names and is often wrong; this asks Arctic
        Shift what actually exists. It is the fallback when nothing the planner
        suggested resolves, which is also what makes the tool usable with no LLM
        key configured.

        Prefix search alone is not enough -- searching "cold" for "cold plunge
        tubs" returns r/Coldplay and r/ColdWarZombies. Candidates are therefore
        scored on how many topic terms appear in the name and description, and
        anything matching only one generic word is dropped.
        """
        words = topic_words(plan.topic)
        if not words:
            return []
        terms = set(words) | {w.rstrip("s") for w in words}

        # Concatenations first: community names are written "coldplunge", not "cold plunge".
        prefixes = ["".join(words), *("".join(p) for p in zip(words, words[1:])), *words]
        seen_prefix: set[str] = set()
        scored: dict[str, tuple[int, int]] = {}

        for pre in prefixes[:6]:
            if pre in seen_prefix:
                continue
            seen_prefix.add(pre)
            try:
                hits = await cl.get("/api/subreddits/search", {"subreddit_prefix": pre, "limit": 40})
            except Exception as e:  # noqa: BLE001
                log.warning("subreddit discovery failed for %r: %s", pre, e)
                continue
            for h in hits:
                name = h.get("display_name")
                subs = int(h.get("subscribers") or 0)
                # Floor drops dead 3-member subs; ceiling keeps the keyword-less
                # sweep meaningful (see SWEEP_MAX_SUBSCRIBERS).
                if not name or not (200 <= subs <= SWEEP_MAX_SUBSCRIBERS):
                    continue
                haystack = f"{name} {h.get('public_description') or ''} {h.get('title') or ''}".lower()
                hay_words = set(re.findall(r"[a-z]{3,}", haystack))
                score = len(terms & (hay_words | {w.rstrip("s") for w in hay_words}))
                # One generic word in common is coincidence (Coldplay vs "cold").
                if score < 2:
                    continue
                prev = scored.get(name)
                if not prev or score > prev[0]:
                    scored[name] = (score, subs)

        top = sorted(scored.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[:want]
        if top:
            log.info("discovered %s", ", ".join(f"r/{n}({v[1]:,})" for n, v in top))
        return [(n, v[1]) for n, v in top]

    async def resolve_subreddits(self, cl: ArcticClient, plan: RunPlan) -> list[tuple[str, int]]:
        """Validate the LLM's guessed subreddits against reality.

        The planner hallucinates plausible-sounding subreddits; this drops the
        ones that don't exist and the ones too big to keyword-search.
        """
        resolved: list[tuple[str, int]] = []
        for name in plan.subreddits:
            n = name.removeprefix("r/").removeprefix("/r/").strip()
            if not n:
                continue
            try:
                # NOTE: this endpoint has no sort-field param -- `sort` only takes
                # asc/desc. Prefix-match then filter for the exact name.
                hits = await cl.get(
                    "/api/subreddits/search", {"subreddit_prefix": n, "limit": 10}
                )
            except Exception as e:  # noqa: BLE001
                log.warning("subreddit resolve failed for %s: %s", n, e)
                continue
            for h in hits:
                dn = h.get("display_name") or ""
                if dn.lower() != n.lower():
                    continue
                subs = h.get("subscribers") or 0
                if subs > TOO_BIG:
                    log.info("skipping r/%s: %s subscribers, too big to keyword-search", dn, subs)
                    break
                resolved.append((dn, int(subs)))
                break
        # de-dup, preserve order
        seen: set[str] = set()
        return [r for r in resolved if not (r[0].lower() in seen or seen.add(r[0].lower()))]

    async def search_posts(
        self, cl: ArcticClient, subreddit: str, keyword: str | None, after: str,
        before: str, limit: int
    ) -> list[Document]:
        """One (subreddit, keyword) slice. `limit` caps at 100 per API call.

        Server-side full-text search is the expensive part of this API and is the
        first thing to fail when Arctic Shift is under load -- it returns
        "Timeout. Maybe slow down a bit" for `query=` while a plain subreddit
        listing of the same range answers instantly. Shrinking the page does not
        help, because the cost is in the text search, not the page size. So when
        a keyword query gives up, re-fetch the slice without it and apply the
        keyword client-side: more bytes over the wire, but the slice is recovered
        instead of lost.
        """
        base = {
            "subreddit": subreddit,
            "after": after,
            "before": before,
            "limit": min(limit, 100),
            "sort": "desc",
        }
        rows = await cl.get("/api/posts/search", {**base, "query": keyword})

        if not rows and keyword:
            rows = await cl.get("/api/posts/search", base)
            if rows:
                log.info(
                    "server-side search for %r in r/%s timed out; filtered %d posts locally",
                    keyword, subreddit, len(rows),
                )
                kw = keyword.lower()
                rows = [
                    r for r in rows
                    if kw in f"{r.get('title') or ''} {r.get('selftext') or ''}".lower()
                ]

        return [d for d in (post_to_doc(r) for r in rows) if d]

    async def fetch_comments(self, cl: ArcticClient, post_id: str, limit: int = 60) -> list[Document]:
        """The comments are where the opinions are; titles are just topics."""
        native = post_id.removeprefix("reddit:")
        rows = await cl.get(
            "/api/comments/search", {"link_id": f"t3_{native}", "limit": min(limit, 100)}
        )
        return [d for d in (comment_to_doc(r) for r in rows) if d]

    async def trend(self, cl: ArcticClient, subreddit: str, after: str, before: str) -> list[tuple[str, int]]:
        """Volume over time straight from the aggregate endpoint -- cheap, and
        doesn't require downloading the posts. Requires before/after to be set."""
        rows = await cl.get(
            "/api/posts/search/aggregate",
            {
                "subreddit": subreddit,
                "aggregate": "created_utc",
                "frequency": "week",
                "after": after,
                "before": before,
                "limit": 200,
            },
        )
        return [(r["created_utc"][:10], int(r["count"])) for r in rows if r.get("created_utc")]

    async def collect(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        before = datetime.now(timezone.utc).date()
        after = before - timedelta(days=plan.days)
        a, b = after.isoformat(), before.isoformat()

        own = self._client is None
        cl = self._client or ArcticClient()
        try:
            subs = await self.resolve_subreddits(cl, plan)
            # A single tiny community that happens to exist is not a corpus, and a
            # planner that guessed one real name out of ten usually guessed badly.
            # Top up from discovery rather than settling for what it produced.
            if len(subs) < 3:
                log.info("only %d subreddit(s) resolved; discovering more from the topic", len(subs))
                known = {n.lower() for n, _ in subs}
                subs += [d for d in await self.discover_subreddits(cl, plan)
                         if d[0].lower() not in known]
            if not subs:
                log.warning("no subreddits found for %r; nothing to collect", plan.topic)
                return []
            log.info(
                "collecting from %d subreddit(s): %s",
                len(subs), ", ".join(f"r/{n}({s:,})" for n, s in subs),
            )

            per_slice = max(10, limit // max(1, len(subs) * (len(plan.keywords[:6]) + 1)))
            posts: dict[str, Document] = {}
            terms = topic_terms(plan)
            dropped = 0

            for sub, subscribers in subs:
                # A keyword-less sweep is how you find what you didn't think to ask
                # about -- but in a large general-interest subreddit it just returns
                # that subreddit's whole front page, which swamps the corpus with
                # off-topic noise. Only sweep communities small enough to be
                # topic-native.
                keywords: list[str | None] = list(plan.keywords[:6])
                if subscribers <= SWEEP_MAX_SUBSCRIBERS:
                    keywords.append(None)
                elif not keywords:
                    log.warning("r/%s is large and no keywords given; skipping", sub)
                    continue

                # In a large general-interest subreddit a generic keyword ("water",
                # "worth it") matches plenty of posts that have nothing to do with
                # the topic, so gate those on a distinctive topic term. Small
                # topic-native subs are exempt: everything there is on-topic by
                # construction, and their posts often omit the topic word entirely
                # ("Washing filters? Who else does it?").
                gate = subscribers > SWEEP_MAX_SUBSCRIBERS
                for kw in keywords:
                    try:
                        found = await self.search_posts(cl, sub, kw, a, b, per_slice)
                    except Exception as e:  # noqa: BLE001
                        log.warning("post search failed r/%s kw=%r: %s", sub, kw, e)
                        continue
                    for d in found:
                        if gate and not is_on_topic(d, terms):
                            dropped += 1
                            continue
                        posts.setdefault(d.id, d)

            if dropped:
                log.info("dropped %d off-topic posts from large subreddits", dropped)
            docs: list[Document] = list(posts.values())

            # Pull comments for the posts with the most discussion under them.
            ranked = sorted(
                posts.values(), key=lambda d: d.engagement.get("num_comments", 0), reverse=True
            )
            targets = [d for d in ranked if d.engagement.get("num_comments", 0) > 1][:60]
            for d in targets:
                try:
                    docs += await self.fetch_comments(cl, d.id)
                except Exception as e:  # noqa: BLE001
                    log.warning("comment fetch failed for %s: %s", d.id, e)

            return docs
        finally:
            if own:
                await cl.__aexit__()
