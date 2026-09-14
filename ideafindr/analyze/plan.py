"""Topic -> RunPlan.

A bare topic ("cold plunge tubs") isn't collectable. Arctic Shift needs
subreddit scopes, the web collector needs queries, and theme analysis needs to
know which brands to count. The LLM supplies all of that; the Reddit collector
then validates the guessed subreddits against the live API, because planners
reliably invent plausible-sounding communities that don't exist.
"""

from __future__ import annotations

from ideafindr.llm import chat_json
from ideafindr.models import RunPlan

SYSTEM = """You are a market research analyst planning a social listening sweep.
You know how people actually talk online: the slang, the abbreviations, the
complaints. You are precise about Reddit community names."""

PROMPT = """Plan a social listening sweep on this topic:

TOPIC: {topic}
TIME WINDOW: last {days} days

Return JSON with these fields:

- "topic": echo the topic
- "days": {days}
- "subreddits": 6-12 real subreddit names (no "r/" prefix) where this topic is
  actually discussed. Prefer specific, topic-native communities over giant
  general ones -- r/AskReddit is useless here, a 12k-member niche sub is gold.
  Include adjacent communities where buyers of this discuss related problems.
- "keywords": 6-10 search terms people genuinely type, including slang,
  abbreviations and misspellings. Favour words that appear in complaints and
  comparisons, not marketing language.
- "brands": the main brands/products/competitors in this space, for share-of-voice
  counting. Empty list if the topic isn't product-shaped.
- "hashtags": 5-8 hashtags without the "#", for TikTok/Instagram if enabled.
- "web_queries": 4-6 open-web search queries likely to surface honest discussion
  (reviews, forum threads, "worth it", "problems with", comparisons).

Be concrete. Generic terms produce a useless corpus."""


def build_plan(topic: str, days: int = 180, model: str | None = None) -> RunPlan:
    plan = chat_json(
        PROMPT.format(topic=topic, days=days), RunPlan, system=SYSTEM, model=model
    )
    # The model drifts on echoed fields; the caller's intent wins.
    plan.topic = topic
    plan.days = days
    plan.subreddits = [s.removeprefix("r/").removeprefix("/r/").strip() for s in plan.subreddits]
    plan.hashtags = [h.lstrip("#").strip() for h in plan.hashtags]
    return plan


def fallback_plan(topic: str, days: int = 180) -> RunPlan:
    """Used when no LLM key is configured, so the pipeline is still testable."""
    words = [w for w in topic.split() if len(w) > 2]
    return RunPlan(
        topic=topic,
        days=days,
        subreddits=["".join(words).lower()] if words else [],
        keywords=words,
        web_queries=[f"{topic} review", f"{topic} problems", f"is {topic} worth it"],
    )
