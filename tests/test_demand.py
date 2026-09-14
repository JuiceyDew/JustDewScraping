"""The demand layer.

These pin the things that would silently produce garbage rather than fail: an
unversioned suggest endpoint changing shape, a coverage measure that matches the
whole corpus, and a gap metric whose sign is backwards.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from ideafindr.demand.analyze import demand_score, score_queries
from ideafindr.demand.gap import _percentiles, compute_gaps, residual_terms
from ideafindr.demand.harvest import MODIFIERS, SUFFIXES, _seed_terms, harvest
from ideafindr.demand.models import DemandCluster, Query
from ideafindr.demand.sources import parse_suggestions


# --- response shapes ----------------------------------------------------------


@pytest.mark.parametrize(
    "payload,expected",
    [
        # google / youtube: [query, [suggestions], [], {metadata}]
        (["cold plunge", ["cold plunge tub", "cold plunge benefits"], [], {}],
         ["cold plunge tub", "cold plunge benefits"]),
        # ddg / bing OpenSearch: [query, [suggestions]]
        (["cold plunge", ["cold plunge tub"]], ["cold plunge tub"]),
        # ddg without type=list has historically returned dicts
        (["cold plunge", [{"phrase": "cold plunge tub"}]], ["cold plunge tub"]),
    ],
)
def test_parse_suggestions_accepts_every_provider_shape(payload, expected):
    assert parse_suggestions(payload) == expected


@pytest.mark.parametrize(
    "payload", [None, {}, [], ["query"], ["query", "not a list"], ["q", [None, 3]]]
)
def test_parse_suggestions_never_raises_on_junk(payload):
    """These endpoints are unversioned and unsupported. A provider changing shape
    must thin the query space, not crash the harvest."""
    assert parse_suggestions(payload) == []


# --- harvesting ---------------------------------------------------------------


def test_seed_terms_cover_alphabet_and_intent_modifiers():
    terms = _seed_terms("cold plunge", SUFFIXES)
    assert "cold plunge" in terms, "the bare topic must be asked too"
    assert "cold plunge a" in terms and "cold plunge z" in terms
    assert "cold plunge worth it" in terms
    assert len(terms) == len(SUFFIXES)
    for m in ("vs", "problems", "alternative", "cheapest"):
        assert m in MODIFIERS, "buying-intent words are the point of the sweep"


@respx.mock
async def test_harvest_dedupes_requests_and_records_rank_and_source():
    calls: list[str] = []

    def handler(request):
        calls.append(str(request.url))
        term = request.url.params.get("q") or request.url.params.get("query") or ""
        return httpx.Response(200, json=[term, [f"{term} tub", f"{term} chiller"], [], {}])

    respx.get(url__startswith="https://suggestqueries.google.com").mock(side_effect=handler)
    respx.get(url__startswith="https://duckduckgo.com").mock(side_effect=handler)
    respx.get(url__startswith="https://api.bing.com").mock(side_effect=handler)

    qs = await harvest("cold plunge", sources=["google"], depth=1, rps=10_000)

    assert qs, "expected suggestions"
    assert {q.source for q in qs} == {"google"}
    assert {q.rank for q in qs} == {0, 1}, "rank is the provider's own ordering"
    # every (source, term) pair is requested at most once
    assert len(calls) == len(set(calls)) == len(SUFFIXES)


@respx.mock
async def test_harvest_survives_a_dead_provider():
    """One endpoint failing must thin the results, never fail the run."""
    respx.get(url__startswith="https://suggestqueries.google.com").mock(
        return_value=httpx.Response(429, text="slow down")
    )
    respx.get(url__startswith="https://duckduckgo.com").mock(
        side_effect=lambda r: httpx.Response(200, json=["x", ["cold plunge tub"]])
    )
    qs = await harvest("cold plunge", sources=["google", "ddg"], depth=1, rps=10_000)
    assert qs and {q.source for q in qs} == {"ddg"}


@respx.mock
async def test_harvest_respects_the_query_budget():
    respx.get(url__startswith="https://suggestqueries.google.com").mock(
        side_effect=lambda r: httpx.Response(200, json=["x", [f"s{i}" for i in range(20)]])
    )
    qs = await harvest("cold plunge", sources=["google"], depth=1, max_queries=25, rps=10_000)
    assert len(qs) == 25


# --- scoring ------------------------------------------------------------------


def test_demand_score_rewards_cross_provider_agreement():
    """Four indexes independently suggesting a query is stronger evidence than
    one suggesting it, which is all the signal there is without volume data."""
    one = [Query(text="q", source="google", seed="s", rank=0)]
    four = [Query(text="q", source=s, seed="s", rank=0)
            for s in ("google", "youtube", "ddg", "bing")]
    assert demand_score(four) > demand_score(one)


def test_demand_score_rewards_higher_rank():
    top = [Query(text="q", source="google", seed="s", rank=0)]
    burried = [Query(text="q", source="google", seed="s", rank=9)]
    assert demand_score(top) > demand_score(burried)


def test_demand_score_rewards_queries_reachable_from_many_seeds():
    one_path = [Query(text="q", source="google", seed="a", rank=0)]
    many = [Query(text="q", source="google", seed=s, rank=0) for s in ("a", "b", "c")]
    assert demand_score(many) > demand_score(one_path)


def test_score_queries_groups_case_and_spacing_variants():
    qs = [
        Query(text="Cold Plunge Tub", source="google", seed="s", rank=0),
        Query(text="cold  plunge tub", source="ddg", seed="s", rank=0),
    ]
    scores = score_queries(qs)
    assert len(scores) == 1, "providers differ on case and spacing for one query"


def test_demand_score_of_nothing_is_zero():
    assert demand_score([]) == 0.0


# --- coverage and gaps --------------------------------------------------------


def test_residual_terms_strips_the_topic():
    """Without this, every query matches most of the corpus on the topic words
    alone -- measuring 'is this about the topic', which everything is."""
    assert residual_terms("cold plunge tub price", "cold plunge tubs") == "price"
    assert residual_terms("cold plunge tub home depot", "cold plunge tubs") == "home depot"


def test_residual_terms_drops_function_words_but_keeps_intent_words():
    r = residual_terms("diy cold plunge tub with chiller", "cold plunge tubs")
    assert r == "diy chiller", "'with' adds nothing to an AND match"
    assert "best" in residual_terms("best cold plunge tub", "cold plunge tubs")
    assert "cheap" in residual_terms("cheap cold plunge tub", "cold plunge tubs")


def test_residual_of_the_bare_topic_is_empty():
    """Nothing distinctive to measure -- callers must not count it as coverage."""
    assert residual_terms("cold plunge tubs", "cold plunge tubs") == ""


def test_percentiles_share_ties():
    assert _percentiles([5.0, 5.0, 5.0]) == [0.5, 0.5, 0.5]
    assert _percentiles([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert _percentiles([7.0]) == [0.5]


def _cluster(cid: int, demand: float, coverage: int) -> DemandCluster:
    return DemandCluster(id=cid, label=f"c{cid}", demand_score=demand, coverage=coverage)


def test_gap_is_positive_when_demand_outruns_coverage():
    """The actionable output: people search it, the corpus cannot speak to it."""
    clusters = compute_gaps([
        _cluster(0, demand=100.0, coverage=0),    # high demand, no content
        _cluster(1, demand=1.0, coverage=100),    # saturated conversation
    ])
    by_id = {c.id: c for c in clusters}
    assert by_id[0].gap > 0
    assert by_id[1].gap < 0
    assert clusters[0].id == 0, "biggest gap must rank first"


def test_gap_is_zero_when_demand_and_coverage_agree():
    clusters = compute_gaps([
        _cluster(0, demand=10.0, coverage=10),
        _cluster(1, demand=20.0, coverage=20),
        _cluster(2, demand=30.0, coverage=30),
    ])
    assert all(c.gap == 0 for c in clusters)


def test_compute_gaps_handles_an_empty_run():
    assert compute_gaps([]) == []


def test_unique_texts_dedupes_across_sources_keeping_best_rank():
    c = DemandCluster(id=0, label="x", queries=[
        Query(text="cold plunge tub", source="google", seed="s", rank=3),
        Query(text="Cold Plunge Tub", source="ddg", seed="s", rank=0),
        Query(text="cold plunge chiller", source="bing", seed="s", rank=1),
    ])
    texts = c.unique_texts()
    assert len(texts) == 2
    assert texts[0] == "Cold Plunge Tub", "best rank across providers wins the ordering"
