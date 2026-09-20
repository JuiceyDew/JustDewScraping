"""Topic -> the query space people actually type.

The mechanic behind AnswerThePublic and TikTok's Creator Search Insights: an
autocomplete endpoint answers with its most-searched completions of a prefix, so
feeding it a topic plus every letter and every intent word recovers far more of
the real query space than the topic alone returns.

    "cold plunge"     -> tub, benefits, near me, chiller, temperature
    "cold plunge c"   -> chiller, cost, cover, cheap
    "cold plunge why" -> why cold plunge after workout, why does it hurt

Budget matters: suffixes x sources x seeds multiplies fast, and these are other
people's endpoints. Depth 1 expands the topic against the full suffix set; depth 2
expands only the most prominent new suggestions, and only with intent modifiers
(the alphabet has already done its work by then).
"""

from __future__ import annotations

import asyncio
import logging
import string
from typing import Callable

import httpx

from ideafindr.collectors.base import AsyncRateLimiter
from ideafindr.config import settings
from ideafindr.demand.models import Query
from ideafindr.demand.sources import ALL_SOURCES, fetch_suggestions

log = logging.getLogger(__name__)

# Question and comparison words are where buying intent lives: "worth it",
# "problems", "vs" and "alternative" surface the doubts a marketer has to answer,
# which the bare topic never returns.
MODIFIERS = [
    "how", "why", "what", "when", "where", "which", "who",
    "is", "are", "can", "does", "do", "should",
    "vs", "versus", "best", "worst", "cheap", "cheapest", "top",
    "alternative", "alternatives", "review", "reviews", "problems", "issues",
    "worth it", "near me", "for", "without", "with", "diy", "cost", "price",
]

SUFFIXES = [""] + list(string.ascii_lowercase) + MODIFIERS

# Progress reporting: (done, total, note). The web UI shows this, replacing
# `collect`'s opaque multi-minute spinner.
ProgressFn = Callable[[int, int, str], None]


def _seed_terms(topic: str, suffixes: list[str]) -> list[str]:
    t = " ".join(topic.split())
    return [f"{t} {s}".strip() if s else t for s in suffixes]


async def harvest(
    topic: str,
    sources: list[str] | None = None,
    depth: int = 2,
    max_queries: int = 1200,
    depth2_seeds: int = 10,
    rps: float | None = None,
    progress: ProgressFn | None = None,
) -> list[Query]:
    """Harvest the query space for a topic across every configured source.

    Returns every (query, source) observation rather than a deduped set: which
    sources agree, and how highly each ranked it, is the entire basis of the
    demand score downstream.
    """
    srcs = sources or list(ALL_SOURCES)
    limiter = AsyncRateLimiter(rps or settings.demand_rps)
    out: list[Query] = []
    seen: set[str] = set()  # (source, term) pairs already requested

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:

        async def one(source: str, term: str, d: int) -> list[Query]:
            await limiter.wait()
            hits = await fetch_suggestions(client, source, term)
            return [
                Query(text=h, source=source, seed=term, depth=d, rank=i)
                for i, h in enumerate(hits)
            ]

        async def sweep(terms: list[str], d: int, note: str) -> list[Query]:
            jobs = [
                (s, t) for t in terms for s in srcs if (s, t) not in seen
            ]
            for job in jobs:
                seen.add(job)
            if not jobs:
                return []
            done = 0
            total = len(jobs)
            found: list[Query] = []
            # Bounded concurrency: the limiter paces requests, this just keeps a
            # few in flight so latency overlaps instead of serialising.
            sem = asyncio.Semaphore(6)

            async def guarded(s: str, t: str) -> list[Query]:
                nonlocal done
                async with sem:
                    res = await one(s, t, d)
                done += 1
                if progress:
                    progress(done, total, f"{note}: {t}")
                return res

            for batch in await asyncio.gather(
                *(guarded(s, t) for s, t in jobs), return_exceptions=True
            ):
                if isinstance(batch, BaseException):
                    log.debug("sweep job failed: %s", batch)
                    continue
                found += batch
            return found

        out += await sweep(_seed_terms(topic, SUFFIXES), 1, "depth 1")
        log.info("depth 1: %d observations from %d source(s)", len(out), len(srcs))

        if depth >= 2 and len(out) < max_queries:
            # Re-seed from the strongest *new* phrases, not the topic again.
            topic_norm = " ".join(topic.lower().split())
            ranked: dict[str, int] = {}
            for q in out:
                if q.key == topic_norm or len(q.key) <= len(topic_norm):
                    continue
                ranked[q.key] = min(ranked.get(q.key, 99), q.rank)
            seeds = [k for k, _ in sorted(ranked.items(), key=lambda kv: kv[1])][:depth2_seeds]
            if seeds:
                terms = [t for s in seeds for t in _seed_terms(s, [""] + MODIFIERS)]
                out += await sweep(terms, 2, "depth 2")
                log.info("depth 2: %d observations total", len(out))

    if len(out) > max_queries:
        # Keep the most prominent observations rather than an arbitrary slice.
        out.sort(key=lambda q: (q.depth, q.rank))
        out = out[:max_queries]

    uniq = {q.key for q in out}
    log.info("harvested %d observations, %d distinct queries", len(out), len(uniq))
    return out
