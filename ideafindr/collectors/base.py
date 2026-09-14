"""Collector protocol.

Every source normalises to Document. Optional/brittle collectors (TikTok,
Instagram) must never fail a run: `safe_collect` swallows their exceptions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

import re

from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Serialise awaiters so they fire at most `rps` times per second.

    Every source this project touches is somebody else's free service -- Arctic
    Shift is one person's, the suggest endpoints are search engines' internal
    ones. Being a good citizen is a correctness concern here, not just courtesy:
    the fastest way to lose a data source is to hammer it.

    Shared by ArcticClient and the demand harvester so there is one implementation
    of the wait rather than one per client.
    """

    def __init__(self, rps: float):
        self._min_interval = 1.0 / max(rps, 0.001)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            gap = self._min_interval - (now - self._last)
            if gap > 0:
                await asyncio.sleep(gap)
            self._last = asyncio.get_event_loop().time()


@runtime_checkable
class Collector(Protocol):
    name: str

    async def collect(self, plan: RunPlan, limit: int) -> list[Document]: ...


async def safe_collect(collector: Collector, plan: RunPlan, limit: int) -> list[Document]:
    """Run a collector; on failure log and return [].

    A dead scraper degrades the corpus, it does not kill the research run.
    """
    try:
        docs = await collector.collect(plan, limit)
        log.info("collector %s returned %d documents", collector.name, len(docs))
        return docs
    except Exception as e:  # noqa: BLE001 - deliberate: isolation is the point
        log.warning("collector %s failed: %s: %s", collector.name, type(e).__name__, e)
        return []


# --- topic gating -------------------------------------------------------------

# Every full-text source has the same failure: a generic keyword matches anything.
# Searching Hacker News for "cold plunge" returns "Stocks Could Plunge 70%";
# Lemmy returns "How do you guys make your coffee?"; r/Biohackers once flooded a
# cold-plunge corpus with 534 posts about boron and peptides. One gate, used by
# every collector that searches a whole site.

TOPIC_STOP = {"the", "and", "for", "with", "best", "top", "new", "how", "why", "what"}



def topic_terms(plan: RunPlan) -> set[str]:
    """Distinctive words that mark a document as being about THIS topic.

    Built from the topic phrase and brand names only -- deliberately not from
    `keywords`, which legitimately include generic words like "water" or
    "worth it" that match anything.
    """
    terms: set[str] = set()
    for src in [plan.topic, *plan.brands]:
        for w in re.findall(r"[a-z]{3,}", src.lower()):
            if w not in TOPIC_STOP:
                terms.add(w)
                terms.add(w.rstrip("s"))  # crude singular
    return terms


def is_on_topic(doc: Document, terms: set[str], min_terms: int = 1) -> bool:
    """Does this document mention enough of the topic to be about it?

    `min_terms` is the whole difference between a scoped and an unscoped source.
    Inside a topic-native subreddit one match is plenty -- everything there is
    on-topic by construction, and posts often omit the topic word entirely
    ("Washing filters? Who else does it?"). Searching a whole site is the
    opposite: for the topic "cold plunge" a single-term match accepts "Stocks
    Could Plunge 70%" and "how do you make your coffee" (cold brew), both of
    which were really returned. Site-wide collectors therefore ask for two.
    """
    if not terms:
        return True
    # Compare stems, not raw terms. topic_terms() stores both "sauna" and
    # "saunas", so counting raw terms makes a one-word topic look like it can
    # supply two -- and min_terms=2 would then reject every document about it.
    stems = {t.rstrip("s") for t in terms}
    words = {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", doc.searchable_text.lower())}
    return len(stems & words) >= min(min_terms, len(stems))


def topic_words(topic: str) -> list[str]:
    """The distinctive words of a topic, for subreddit discovery."""
    return [w for w in re.findall(r"[a-z]{3,}", topic.lower()) if w not in TOPIC_STOP]


