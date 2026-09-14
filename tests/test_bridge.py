"""The bridge is the contract gpt-researcher depends on. If these break, reports
silently fall back to web-only sources and stop citing the social corpus.
"""

from __future__ import annotations

import pytest

import ideafindr.config as cfg
from ideafindr.bridge import retriever_api
from ideafindr.embed import TfidfEmbedder
from ideafindr.store import db
from tests.conftest import make_doc


@pytest.fixture
def corpus(tmp_path, plan, monkeypatch):
    monkeypatch.setattr(cfg.settings, "db_path", tmp_path / "t.db")
    retriever_api._state["embedder"] = TfidfEmbedder()
    retriever_api._state["vectors"] = {}

    con = db.connect(tmp_path / "t.db")
    db.create_run(con, "r1", plan, ["reddit"])
    db.save_documents(con, "r1", [
        make_doc(1, "it hums all night and the noise is unbearable",
                 kind="post", title="Chiller noise is driving me mad",
                 engagement={"score": 20, "num_comments": 3}),
        make_doc(2, "mine was the same until I moved it into the garage",
                 kind="comment", parent_id="reddit:1", engagement={"score": 9}),
        make_doc(3, "put it on a rubber mat, cut the vibration right down",
                 kind="comment", parent_id="reddit:1", engagement={"score": 15}),
        make_doc(4, "my water keeps going cloudy after a week of daily use",
                 kind="post", title="Water quality problems", engagement={"score": 5}),
    ])
    con.close()
    yield "r1"
    retriever_api._state["vectors"] = {}


def test_response_matches_gpt_researcher_custom_retriever_schema(corpus):
    res = retriever_api.search_corpus(query="chiller noise", run_id=corpus, limit=3)
    assert isinstance(res, list) and res
    for item in res:
        assert set(item) == {"url", "raw_content"}, "upstream reads exactly these keys"
        assert item["url"].startswith("https://")
        assert isinstance(item["raw_content"], str) and item["raw_content"]


def test_comments_resolve_to_their_parent_thread(corpus):
    """A lone comment gives the agent nothing to judge or attribute, so a matching
    comment must be served as the whole thread it belongs to."""
    res = retriever_api.search_corpus(query="rubber mat vibration garage", run_id=corpus, limit=2)
    top = res[0]["raw_content"]
    assert "Chiller noise is driving me mad" in top
    assert "--- comments ---" in top
    assert "rubber mat" in top


def test_threads_are_not_returned_twice(corpus):
    res = retriever_api.search_corpus(query="noise garage mat hums", run_id=corpus, limit=5)
    assert len({r["url"] for r in res}) == len(res)


def test_unknown_run_returns_empty_not_error(corpus):
    assert retriever_api.search_corpus(query="anything", run_id="does-not-exist") == []


def test_rrf_rewards_agreement_between_rankings():
    fused = retriever_api.rrf(["a", "b", "c"], ["c", "a", "z"])
    assert fused[0] == "a", "a ranks highly in both lists"
    assert set(fused) == {"a", "b", "c", "z"}
