"""The cookie-based collectors and the batched LLM helper.

These pin the behaviours that broke during the rework: PEP 695 aliases are not
callable, site-wide searches need the two-term topic gate, and batched labelling
must tolerate a partial answer.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import ideafindr.llm as llm
from ideafindr.llm import _Batch, _BatchItem
from ideafindr.models import Document, RunPlan


# --- platform aliases -----------------------------------------------------------


def test_new_platforms_validate_on_document():
    """Platform is a TypeAliasType (not callable), but its literal values must be
    accepted by Document. This is what silently broke every new collector."""
    for platform in ("bluesky", "github", "mastodon", "producthunt", "x"):
        d = Document(
            id=f"{platform}:1", platform=platform, kind="post",
            url="https://example.com", text="hello",
            created_at=datetime.now(timezone.utc),
        )
        assert d.platform == platform


def test_unknown_platform_is_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Document(
            id="nope:1", platform="myspace", kind="post",
            url="u", text="t", created_at=datetime.now(timezone.utc),
        )


# --- batched labelling ----------------------------------------------------------


def test_batch_splits_and_aligns(monkeypatch):
    calls = []

    def fake(prompt, schema, **kw):
        import re

        idxs = [int(m) for m in re.findall(r"### Cluster (\d+)", prompt)]
        calls.append(len(idxs))
        return _Batch(items=[_BatchItem(index=i, label=f"L{i}") for i in idxs])

    monkeypatch.setattr(llm, "chat_json", fake)
    out = llm.chat_json_batch(["x"] * 19, system="s", instructions="i",
                              max_blocks_per_call=12)
    assert len(out) == 19
    assert calls == [12, 7]
    assert [o.label for o in out] == [f"L{i}" for i in range(12)] + [f"L{i}" for i in range(7)]


def test_batch_tolerates_a_partial_answer(monkeypatch):
    """A missing item must be None, not shift every later label."""

    def partial(prompt, schema, **kw):
        import re

        idxs = [int(m) for m in re.findall(r"### Cluster (\d+)", prompt)]
        return _Batch(items=[_BatchItem(index=i, label="ok") for i in idxs if i != 1])

    monkeypatch.setattr(llm, "chat_json", partial)
    out = llm.chat_json_batch(["x"] * 3, system="s", instructions="i")
    assert out[0] is not None and out[1] is None and out[2] is not None


def test_batch_failure_returns_all_none(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(llm, "chat_json", boom)
    out = llm.chat_json_batch(["x"] * 3, system="s", instructions="i")
    assert out == [None, None, None]


# --- X collector ----------------------------------------------------------------


def test_x_tweet_maps_to_document():
    from dataclasses import dataclass

    from ideafindr.collectors.x_twitter import _tweet_to_doc

    @dataclass
    class User:
        username: str = "someone"
        displayname: str = "Some One"

    @dataclass
    class Tweet:
        id: int = 1
        id_str: str = "1"
        rawContent: str = "chiller noise is unreal"
        date: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)
        url: str = "https://x.com/someone/status/1"
        user: object = None
        hashtags: object = None
        likeCount: int = 5
        retweetCount: int = 1
        replyCount: int = 2
        quoteCount: int = 0
        viewCount: int = 100
        lang: str = "en"

    t = Tweet()
    t.user = User()
    t.hashtags = ["coldplunge"]
    d = _tweet_to_doc(t)
    assert d.id == "x:1" and d.platform == "x" and d.community == "#coldplunge"
    assert d.engagement["likes"] == 5 and d.engagement["reposts"] == 1


def test_x_queries_exclude_retweets_and_cap():
    from ideafindr.collectors.x_twitter import _queries

    plan = RunPlan(topic="cold plunge tubs",
                   keywords=["chiller", "water", "cost", "noise", "brand"],
                   brands=["Ice Barrel", "Plunge"])
    qs = _queries(plan, 4)
    assert len(qs) == 4
    assert all("-filter:retweets" in q for q in qs)
    assert qs[0].startswith("cold plunge tubs")


def test_x_collect_returns_empty_without_credentials(monkeypatch):
    import asyncio

    from ideafindr.collectors import x_twitter

    monkeypatch.setattr(x_twitter, "has_credentials", lambda: False)
    out = asyncio.run(x_twitter.XCollector().collect(RunPlan(topic="x"), 10))
    assert out == []


# --- Instagram --------------------------------------------------------------------


def test_instagram_has_session_detects_config(monkeypatch):
    from ideafindr.collectors import instagram

    monkeypatch.setattr(instagram.settings, "instagram_sessionid", "")
    monkeypatch.setattr(instagram.settings, "instagram_session_user", "")
    assert instagram.has_session() is False

    monkeypatch.setattr(instagram.settings, "instagram_sessionid", "abc")
    assert instagram.has_session() is True


# --- Mastodon HTML stripping ------------------------------------------------------


def test_mastodon_strips_html_and_entities():
    from ideafindr.collectors.mastodon import MastodonCollector

    d = MastodonCollector()._parse_post(
        {"id": "1", "content": "<p>cold plunge &amp; sauna</p>",
         "created_at": "2024-01-01T00:00:00Z", "account": {"acct": "x"}, "url": "u"},
        "https://mastodon.social",
    )
    assert d is not None
    assert "<p>" not in d.text and "&amp;" not in d.text
    assert d.text == "cold plunge sauna"
    assert d.id == "mastodon:mastodon.social:1"
