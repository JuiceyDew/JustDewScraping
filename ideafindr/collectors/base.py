"""Collector protocol.

Every source normalises to Document. Optional/brittle collectors (TikTok,
Instagram) must never fail a run: `safe_collect` swallows their exceptions.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)


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
