"""Core data model.

Everything downstream depends on `Document` being right: one schema, every platform.
Collectors normalise into it, the store persists it, the analysis layer clusters it,
and the bridge serves it to gpt-researcher.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Python 3.12+ type aliases for cleaner type hints.
#
# These are `TypeAliasType`, which is NOT callable -- collectors must pass plain
# string literals to Document (platform="bluesky"), never Platform("bluesky").
type Platform = Literal[
    "reddit", "reddit-search", "web", "hackernews", "lemmy", "stackexchange",
    "tiktok", "instagram", "x", "bluesky", "github", "mastodon", "producthunt",
]
type Kind = Literal["post", "comment", "article", "video", "caption"]
type Stance = Literal["pain_point", "desire", "objection", "praise", "question", "neutral"]


class Document(BaseModel):
    """A single piece of what someone said, from any platform."""

    id: str  # f"{platform}:{native_id}" -- primary dedup key
    platform: Platform
    kind: Kind
    url: str  # permalink; reports cite these, so it must resolve
    title: str | None = None
    text: str = ""
    author: str | None = None
    created_at: datetime
    parent_id: str | None = None  # comment -> post threading
    community: str | None = None  # subreddit / hashtag / domain
    engagement: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        """Naive datetimes silently corrupt every time-series signal downstream."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @property
    def searchable_text(self) -> str:
        """Title + body, for embedding and keyword indexing."""
        return f"{self.title}\n\n{self.text}".strip() if self.title else self.text

    def engagement_score(self) -> float:
        """Cross-platform 'how much did this resonate' number.

        Deliberately crude: platforms count attention differently, so this is only
        ever used to rank *within* a platform or to weight a theme's volume.
        """
        e = self.engagement
        return float(
            e.get("score", 0)
            + 2 * e.get("num_comments", 0)
            + e.get("likes", 0) / 10
            + e.get("views", 0) / 1000
        )


class RunPlan(BaseModel):
    """The LLM's expansion of a bare topic into something collectable."""

    topic: str
    subreddits: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    brands: list[str] = Field(default_factory=list)
    web_queries: list[str] = Field(default_factory=list)
    # Stack Exchange sites worth searching, e.g. ["diy", "physics"]. Empty for
    # most consumer topics, which is why the collector skips rather than guesses.
    stack_sites: list[str] = Field(default_factory=list)
    days: int = 180

    def summary(self) -> str:
        return (
            f"topic={self.topic!r} days={self.days} "
            f"subreddits={len(self.subreddits)} keywords={len(self.keywords)} "
            f"brands={len(self.brands)}"
        )


class Theme(BaseModel):
    """A cluster of documents that are about the same thing, named by the LLM."""

    id: int
    label: str
    description: str = ""
    stance: Stance = "neutral"
    doc_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    # False when the labeller judged the cluster to be incoherent -- clustering
    # always returns k groups whether or not k real themes exist, so there is
    # usually one bucket of leftovers. Kept (the documents are real) but ranked
    # last and excluded from the report body.
    coherent: bool = True

    # signals, filled in by analyze/signals.py
    volume: int = 0
    engagement_weighted: float = 0.0
    momentum: float = 0.0  # recent share vs. baseline share; >1 means rising
    trend: list[tuple[str, int]] = Field(default_factory=list)  # [(week_iso, count)]
    quotes: list[Quote] = Field(default_factory=list)


class Quote(BaseModel):
    """A verbatim. Never paraphrased -- the verbatim IS the deliverable."""

    text: str
    url: str
    author: str | None = None
    community: str | None = None
    created_at: datetime
    engagement: float = 0.0


class Run(BaseModel):
    """One research run, start to finish."""

    id: str
    topic: str
    created_at: datetime
    plan: RunPlan
    doc_count: int = 0
    sources: list[str] = Field(default_factory=list)


Theme.model_rebuild()
