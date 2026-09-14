"""Collector protocol.

Every source normalises to Document. Optional/brittle collectors (TikTok,
Instagram) must never fail a run: `safe_collect` swallows their exceptions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

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
