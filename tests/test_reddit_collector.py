"""Arctic Shift collector, against recorded responses.

These pin the two live-API behaviours that actually broke things during
development: a 200 response carrying `"data": null`, and a 400/422 query error.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from ideafindr.collectors.reddit_arctic import (
    ArcticClient,
    RedditArcticCollector,
    comment_to_doc,
    is_bot,
    is_on_topic,
    post_to_doc,
    topic_terms,
)
from ideafindr.models import RunPlan

BASE = "https://arctic-shift.photon-reddit.com"

RAW_POST = {
    "id": "abc123", "title": "Noisy chiller", "selftext": "it hums all night",
    "author": "someone", "created_utc": 1789168122, "score": 12, "num_comments": 8,
    "subreddit": "coldplunge", "upvote_ratio": 0.95,
    "permalink": "/r/coldplunge/comments/abc123/noisy_chiller/",
    "url": "https://i.redd.it/whatever.jpeg",
}
RAW_COMMENT = {
    "id": "c1", "body": "mine does the same", "author": "other",
    "created_utc": 1789168200, "score": 3, "link_id": "t3_abc123",
    "subreddit": "coldplunge", "permalink": "/r/coldplunge/comments/abc123/x/c1/",
}


def test_post_permalink_is_preferred_over_link_target():
    """`url` on a link post points at the image, not the discussion. Reports cite
    the discussion."""
    d = post_to_doc(RAW_POST)
    assert d.url == "https://www.reddit.com/r/coldplunge/comments/abc123/noisy_chiller/"
    assert d.kind == "post" and d.community == "coldplunge"
    assert d.engagement["num_comments"] == 8


def test_comment_links_to_parent_post():
    d = comment_to_doc(RAW_COMMENT)
    assert d.parent_id == "reddit:abc123" and d.kind == "comment"


@pytest.mark.parametrize("body", ["[deleted]", "[removed]", ""])
def test_removed_comments_are_dropped(body):
    assert comment_to_doc({**RAW_COMMENT, "body": body}) is None


def test_moderator_bots_are_dropped():
    assert is_bot("AutoModerator", "rules")
    assert is_bot("someone", "I am a bot, beep boop")
    assert not is_bot("someone", "a genuine opinion about chillers")
    assert comment_to_doc({**RAW_COMMENT, "author": "AutoModerator"}) is None


def test_topic_gate_excludes_offtopic_but_keeps_brands():
    terms = topic_terms(RunPlan(topic="cold plunge tubs", brands=["Ice Barrel"]))
    on = post_to_doc({**RAW_POST, "title": "my plunge water is cloudy"})
    off = post_to_doc({**RAW_POST, "title": "boron killed my sleep", "selftext": "magnesium"})
    brand = post_to_doc({**RAW_POST, "title": "the Ice Barrel is great", "selftext": ""})
    assert is_on_topic(on, terms) and is_on_topic(brand, terms)
    assert not is_on_topic(off, terms)


@respx.mock
async def test_data_null_is_retried_not_treated_as_empty():
    """A 200 with data:null means throttled. Treating it as an empty result
    silently loses whole slices of the corpus."""
    route = respx.get(f"{BASE}/api/posts/search").mock(
        side_effect=[
            httpx.Response(200, json={"data": None}),
            httpx.Response(200, json={"data": [RAW_POST]}),
        ]
    )
    async with ArcticClient(rps=1000) as cl:
        rows = await cl.get("/api/posts/search", {"subreddit": "coldplunge", "limit": 100})
    assert route.call_count == 2
    assert rows[0]["id"] == "abc123"


@respx.mock
async def test_query_timeout_retries_with_a_smaller_page():
    """"Timeout. Maybe slow down a bit" means the query was too expensive --
    the fix is a smaller page, not a longer wait."""
    seen: list[str] = []

    def handler(request):
        limit = request.url.params["limit"]
        seen.append(limit)
        if int(limit) > 25:
            return httpx.Response(422, json={"data": None, "error": "Timeout. Maybe slow down a bit"})
        return httpx.Response(200, json={"data": [RAW_POST]})

    respx.get(f"{BASE}/api/posts/search").mock(side_effect=handler)
    async with ArcticClient(rps=1000) as cl:
        rows = await cl.get("/api/posts/search", {"subreddit": "coldplunge", "limit": 100})
    assert seen == ["100", "50", "25"]
    assert len(rows) == 1


@respx.mock
async def test_malformed_query_is_not_retried():
    route = respx.get(f"{BASE}/api/posts/search").mock(
        return_value=httpx.Response(400, json={"data": None, "error": "'query' requires one of: author, subreddit"})
    )
    async with ArcticClient(rps=1000) as cl:
        assert await cl.get("/api/posts/search", {"query": "x"}) == []
    assert route.call_count == 1


@respx.mock
async def test_resolve_subreddits_drops_fakes_and_giants():
    respx.get(f"{BASE}/api/subreddits/search").mock(side_effect=lambda r: httpx.Response(
        200,
        json={"data": {
            "coldplunge": [{"display_name": "coldplunge", "subscribers": 13000}],
            "nosuchsub": [],
            "AskReddit": [{"display_name": "AskReddit", "subscribers": 51_000_000}],
        }[r.url.params["subreddit_prefix"]]},
    ))
    plan = RunPlan(topic="cold plunge", subreddits=["r/coldplunge", "nosuchsub", "AskReddit"])
    async with ArcticClient(rps=1000) as cl:
        got = await RedditArcticCollector().resolve_subreddits(cl, plan)
    assert got == [("coldplunge", 13000)]


@respx.mock
async def test_keyword_search_timeout_falls_back_to_local_filtering():
    """Server-side full-text search is the first thing to fail under load, and
    shrinking the page doesn't help because the cost is the text search. The
    slice must be recovered by fetching unfiltered and matching locally."""
    def handler(request):
        if "query" in request.url.params:
            return httpx.Response(422, json={"data": None, "error": "Timeout. Maybe slow down a bit"})
        return httpx.Response(200, json={"data": [
            {**RAW_POST, "id": "hit", "title": "my chiller is loud"},
            {**RAW_POST, "id": "miss", "title": "water is cloudy", "selftext": ""},
        ]})

    respx.get(f"{BASE}/api/posts/search").mock(side_effect=handler)
    async with ArcticClient(rps=1000) as cl:
        docs = await RedditArcticCollector().search_posts(
            cl, "coldplunge", "chiller", "2026-01-01", "2026-06-01", 100
        )
    assert [d.id for d in docs] == ["reddit:hit"], "only the keyword match survives"
