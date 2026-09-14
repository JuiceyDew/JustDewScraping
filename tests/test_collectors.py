"""The newer collectors, against recorded responses.

These pin the things that actually bite: Reddit's `{"kind","data"}` envelope, an
expired OAuth token, HN comments having no score field, Lemmy's cross-instance
duplicates, Stack Exchange's tiny daily quota, and the topic gate that keeps
site-wide search from filling a corpus with whatever shares a common word.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from ideafindr.collectors.base import is_on_topic, topic_terms
from ideafindr.collectors.dork import DorkCollector, _doc as dork_doc
from ideafindr.collectors.hackernews import HackerNewsCollector, _doc as hn_doc
from ideafindr.collectors.lemmy import LemmyCollector, post_to_doc as lemmy_post
from ideafindr.collectors.reddit_api import (
    RedditAPIClient,
    RedditAPICollector,
    RedditAuthError,
)
from ideafindr.collectors.reddit_common import unwrap
from ideafindr.collectors.stackexchange import MAX_REQUESTS_PER_RUN, StackExchangeCollector
from ideafindr.models import Document, RunPlan
from datetime import datetime, timezone


def _doc(text: str, title: str | None = None) -> Document:
    return Document(id="x:1", platform="web", kind="post", url="https://x",
                    title=title, text=text, created_at=datetime.now(timezone.utc))


# --- the topic gate -----------------------------------------------------------


def test_one_matching_term_is_enough_inside_a_scoped_source():
    """In a topic-native subreddit everything is on-topic by construction, and
    posts often omit the topic word ("Washing filters? Who else does it?")."""
    terms = topic_terms(RunPlan(topic="cold plunge"))
    assert is_on_topic(_doc("my plunge water is cloudy"), terms) is True


def test_site_wide_search_needs_two_terms():
    """Real results from Hacker News and Lemmy for the topic "cold plunge". One
    common word in common is a coincidence, not a match."""
    terms = topic_terms(RunPlan(topic="cold plunge"))
    stocks = _doc("", "Grantham Warns U.S. Stocks Could Plunge 70%")
    coffee = _doc("I use a cold brew concentrate", "How do you guys make your coffee?")
    real = _doc("", "Why Saunas, Bathhouses, Cold Plunge Pools Are So Popular")

    assert is_on_topic(stocks, terms, min_terms=2) is False
    assert is_on_topic(coffee, terms, min_terms=2) is False
    assert is_on_topic(real, terms, min_terms=2) is True


def test_single_word_topic_cannot_be_gated_to_two_terms():
    """min_terms must never exceed what the topic can supply, or a one-word topic
    would reject everything."""
    terms = topic_terms(RunPlan(topic="saunas"))
    assert is_on_topic(_doc("my sauna is great"), terms, min_terms=2) is True


# --- reddit api ---------------------------------------------------------------


def test_reddit_api_unwraps_the_kind_data_envelope():
    """Reddit wraps every item; Arctic Shift does not. One normaliser serves both
    only because of this."""
    assert unwrap({"kind": "t3", "data": {"id": "abc"}}) == {"id": "abc"}
    assert unwrap({"id": "abc"}) == {"id": "abc"}, "Arctic Shift's bare shape"


async def test_reddit_api_without_credentials_says_so_clearly():
    cl = RedditAPIClient(client_id="", client_secret="")
    with pytest.raises(RedditAuthError, match="REDDIT_CLIENT_ID"):
        await cl.token()
    await cl.__aexit__()


@respx.mock
async def test_reddit_api_caches_its_token_and_refreshes_when_stale():
    """One token per run, not one per request -- the rate limit is the scarce
    resource and an extra auth round trip on every call wastes it."""
    route = respx.post("https://www.reddit.com/api/v1/access_token").mock(
        side_effect=[
            httpx.Response(200, json={"access_token": "first", "expires_in": 3600}),
            httpx.Response(200, json={"access_token": "second", "expires_in": 3600}),
        ]
    )
    cl = RedditAPIClient(client_id="id", client_secret="secret", rps=10_000)
    assert await cl.token() == "first"
    assert await cl.token() == "first", "cached"
    assert route.call_count == 1

    cl._expires_at = 0  # simulate expiry
    assert await cl.token() == "second"
    assert route.call_count == 2
    await cl.__aexit__()


@respx.mock
async def test_reddit_api_rejects_bad_credentials_without_retrying():
    route = respx.post("https://www.reddit.com/api/v1/access_token").mock(
        return_value=httpx.Response(401, json={"error": "invalid_client"})
    )
    cl = RedditAPIClient(client_id="bad", client_secret="bad", rps=10_000)
    with pytest.raises(RedditAuthError, match="rejected the credentials"):
        await cl.token()
    assert route.call_count == 1, "wrong credentials fail identically on a retry"
    await cl.__aexit__()


@respx.mock
async def test_reddit_api_search_normalises_into_documents():
    respx.post("https://www.reddit.com/api/v1/access_token").mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
    )
    respx.get(url__startswith="https://oauth.reddit.com/r/coldplunge/search").mock(
        return_value=httpx.Response(200, json={"data": {"children": [
            {"kind": "t3", "data": {
                "id": "abc123", "title": "Noisy chiller", "selftext": "it hums",
                "author": "someone", "created_utc": 1789168122, "score": 12,
                "num_comments": 8, "subreddit": "coldplunge",
                "permalink": "/r/coldplunge/comments/abc123/noisy_chiller/",
            }},
        ]}})
    )
    cl = RedditAPIClient(client_id="id", client_secret="s", rps=10_000)
    docs = await RedditAPICollector().search_posts(cl, "coldplunge", "chiller", 25)
    await cl.__aexit__()

    assert len(docs) == 1
    d = docs[0]
    assert d.id == "reddit:abc123" and d.platform == "reddit"
    assert d.url == "https://www.reddit.com/r/coldplunge/comments/abc123/noisy_chiller/"
    assert d.engagement["score"] == 12


@respx.mock
async def test_reddit_api_survives_a_rate_limit_without_failing_the_run():
    respx.post("https://www.reddit.com/api/v1/access_token").mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
    )
    respx.get(url__startswith="https://oauth.reddit.com/").mock(
        return_value=httpx.Response(429, headers={"x-ratelimit-reset": "0"}, json={})
    )
    cl = RedditAPIClient(client_id="id", client_secret="s", rps=10_000)
    assert await cl.listing("/r/x/search", {"q": "y"}) == []
    await cl.__aexit__()


# --- hacker news --------------------------------------------------------------


def test_hn_comments_inherit_their_story_score():
    """HN comment hits carry no `points` field at all. Scoring them 0 would sink
    every HN comment to the bottom of quote selection."""
    hit = {"objectID": "9", "created_at_i": 1789168122, "story_id": "1",
           "comment_text": "x" * 120, "author": "someone"}
    assert hn_doc(hit, "comment", story_score=0).engagement["score"] == 0
    assert hn_doc(hit, "comment", story_score=57).engagement["score"] == 57


def test_hn_drops_fragment_comments():
    hit = {"objectID": "9", "created_at_i": 1789168122, "comment_text": "same here"}
    assert hn_doc(hit, "comment") is None


def test_hn_story_permalink_points_at_the_discussion():
    """The `url` field is the linked article; the discussion is the deliverable."""
    hit = {"objectID": "42", "created_at_i": 1789168122, "title": "Cold plunge",
           "points": 9, "url": "https://example.com/article"}
    assert hn_doc(hit, "story").url == "https://news.ycombinator.com/item?id=42"


# --- lemmy --------------------------------------------------------------------


def test_lemmy_dedupes_on_ap_id_across_instances():
    """Federation means one post has a different local id on every instance.
    ap_id is its identity everywhere."""
    view = {
        "post": {"ap_id": "https://lemmy.world/post/1", "name": "Cold plunge",
                 "body": "b", "published": "2026-05-01T10:00:00Z", "id": 1},
        "counts": {"score": 5, "comments": 2},
        "creator": {"name": "someone"}, "community": {"name": "fitness"},
    }
    a = lemmy_post({**view, "_instance": "lemmy.world"})
    b = lemmy_post({**view, "post": {**view["post"], "id": 987},
                    "_instance": "lemmy.ml"})
    assert a.id == b.id, "same post seen from two instances is one document"


def test_lemmy_skips_removed_posts():
    view = {"post": {"ap_id": "x", "name": "t", "published": "2026-05-01T10:00:00Z",
                     "removed": True}, "counts": {}, "creator": {}, "community": {}}
    assert lemmy_post(view) is None


# --- stack exchange -----------------------------------------------------------


async def test_stackexchange_skips_entirely_when_no_sites_chosen():
    """The planner returns no sites for most consumer topics. Spending the 300/day
    quota to discover that is the wrong trade."""
    got = await StackExchangeCollector().collect(RunPlan(topic="cold plunge tubs"), 50)
    assert got == []


@respx.mock
async def test_stackexchange_caps_requests_per_run():
    """300 requests/day unkeyed. One run must not be able to spend the lot."""
    route = respx.get(url__startswith="https://api.stackexchange.com").mock(
        return_value=httpx.Response(200, json={"items": [], "quota_remaining": 250})
    )
    plan = RunPlan(topic="raspberry pi cooling",
                   keywords=["fan", "heatsink", "thermal"],
                   stack_sites=["raspberrypi", "electronics", "unix"])
    await StackExchangeCollector(rps=10_000).collect(plan, 500)
    assert route.call_count <= MAX_REQUESTS_PER_RUN


# --- dork ---------------------------------------------------------------------


def test_dork_only_accepts_real_thread_urls():
    now = datetime.now(timezone.utc)
    body = "I paid like $4,000 for it and I can't get over how loud it is, honestly"
    good = dork_doc({"href": "https://www.reddit.com/r/coldplunge/comments/abc123/x/",
                     "title": "So loud", "body": body}, now)
    assert good and good.community == "coldplunge"
    assert good.platform == "reddit-search"
    assert dork_doc({"href": "https://www.reddit.com/r/coldplunge/", "body": body}, now) is None
    assert dork_doc({"href": "https://example.com/blog", "body": body}, now) is None


def test_dork_ids_never_collide_with_fully_collected_reddit_posts():
    """A snippet must not dedupe away the full thread if both are collected."""
    now = datetime.now(timezone.utc)
    d = dork_doc({"href": "https://www.reddit.com/r/coldplunge/comments/abc123/x/",
                  "title": "t", "body": "y" * 80}, now)
    assert d.id == "redditsearch:abc123" and d.id != "reddit:abc123"


def test_dork_documents_are_marked_undated():
    """Search results have no publication date. Counted in the time series they
    would put the whole corpus in the current week."""
    from ideafindr.analyze.signals import UNDATED_PLATFORMS

    assert "reddit-search" in UNDATED_PLATFORMS


def test_dork_queries_target_opinion_not_product_pages():
    qs = DorkCollector().queries(RunPlan(topic="cold plunge tubs", keywords=["chiller"]))
    assert all(q.startswith("site:reddit.com") for q in qs)
    joined = " ".join(qs)
    assert "problems" in joined and "worth it" in joined
