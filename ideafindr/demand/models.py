"""Demand-side data model.

Deliberately separate from `ideafindr.models.Document`, which is "a single piece
of what someone said" and requires a resolving permalink, a timestamp, an author
and engagement counts. A search query has none of those -- it is a string, a
source and a position in a suggestion list. Forcing queries into Document would
corrupt every corpus statistic (corpus_stats, language_bank and share_of_voice
all assume utterances).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QuerySource = Literal["google", "youtube", "ddg", "bing"]


class Query(BaseModel):
    """One autocomplete suggestion, exactly as the provider spelled it."""

    text: str
    source: QuerySource
    seed: str  # the expansion that surfaced it
    depth: int = 1  # 1 = expanded directly from the topic
    rank: int = 0  # position in the provider's suggestion list, 0 = first

    @property
    def key(self) -> str:
        """Dedup key. Providers differ on case and spacing for the same query."""
        return " ".join(self.text.lower().split())


class DemandCluster(BaseModel):
    """A group of queries expressing one search intent."""

    id: int
    label: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    queries: list[Query] = Field(default_factory=list)

    # See demand/analyze.py::demand_score for what this number is and is not.
    demand_score: float = 0.0
    sources: dict[str, int] = Field(default_factory=dict)

    # Filled in by demand/gap.py -- the join back to the conversation corpus.
    theme_id: int | None = None
    theme_label: str = ""
    coverage: int = 0
    gap: float = 0.0

    @property
    def size(self) -> int:
        return len(self.queries)

    def unique_texts(self) -> list[str]:
        """Query texts, deduped across sources, most prominent first."""
        best: dict[str, Query] = {}
        for q in self.queries:
            prev = best.get(q.key)
            if prev is None or q.rank < prev.rank:
                best[q.key] = q
        return [q.text for q in sorted(best.values(), key=lambda q: q.rank)]
