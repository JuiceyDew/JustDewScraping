"""Join demand to conversation, and find the gaps.

A keyword tool tells you people search "cold plunge chiller noise". It cannot
tell you why, because it has never read a word anyone wrote. This project has a
corpus with permalinks and verbatim quotes, so each demand cluster can be matched
back to the conversation that explains it -- and, more usefully, to the absence of
one.

Retrieval reuses what the bridge already does: FTS5 keyword matching fused with
vector similarity by reciprocal rank. There is no reason for a third ranker.
"""

from __future__ import annotations

import logging
import re
import sqlite3

import numpy as np

from ideafindr.bridge.retriever_api import rrf
from ideafindr.demand.models import DemandCluster
from ideafindr.embed import Embedder, get_embedder
from ideafindr.models import Theme
from ideafindr.store import db

log = logging.getLogger(__name__)

# Cap on documents credited to one cluster. Past a point more documents say
# nothing further about whether you can speak to this demand, and an unbounded
# count lets one very common word dominate every percentile.
MAX_COVERAGE = 100

# How many documents to count per query when measuring coverage. This is a
# counting operation, not a top-k retrieval: the question is how much of the
# corpus speaks to a query, not which single document answers it best.
COVERAGE_LIMIT = 400

# Below this many matching documents, a theme "explaining" a demand cluster is
# coincidence rather than signal.
MIN_THEME_VOTES = 3

# Platforms whose documents are long-form articles rather than utterances. A
# 20,000-character review page mentions nearly everything once, so it matches
# almost any query and is near-worthless as evidence that a *conversation* covers
# a search intent -- it swamped the theme vote entirely until excluded (every
# demand cluster mapped to the one web-heavy theme). Such documents still count
# in the corpus, in themes and in the language bank; they just do not get to
# decide which conversation explains a demand cluster.
LONGFORM_PLATFORMS = {"web"}

# Function words only. These are dropped from a residual because an AND match on
# "with" tells you nothing, and their presence turns a precise query into a broad
# one. Intent words -- best, cheap, diy, how, why, worth -- are deliberately NOT
# here: they are exactly what distinguishes one search intent from another.
_FUNCTION_WORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "these", "those",
    "your", "you", "their", "are", "was", "were", "been", "being", "have",
    "has", "had", "does", "did", "will", "would", "could", "should", "any", "all",
}


def residual_terms(query: str, topic: str) -> str:
    """The part of a query that is NOT just the topic restated.

    This matters more than it looks. db.search_fts joins terms with OR, so
    searching a cold-plunge corpus for "cold plunge tub price" matches 39% of it
    on the topic words alone -- measuring "is this about cold plunges", which
    every document is. Stripping the topic leaves "price", which actually
    discriminates: that is the difference between a coverage number and noise.

    Returns "" when a query is nothing but the topic, which callers read as
    "no distinctive demand to measure".
    """
    topic_tokens = set(re.findall(r"[a-z0-9]+", topic.lower()))
    topic_tokens |= {t.rstrip("s") for t in topic_tokens}
    out = [
        w
        for w in re.findall(r"[a-z0-9]+", query.lower())
        if len(w) > 2
        and w not in topic_tokens
        and w.rstrip("s") not in topic_tokens
        and w not in _FUNCTION_WORDS
    ]
    return " ".join(out)


def _theme_of(themes: list[Theme]) -> dict[str, int]:
    """doc_id -> theme id. A document belongs to at most one theme."""
    out: dict[str, int] = {}
    for t in themes:
        for d in t.doc_ids:
            out[d] = t.id
    return out


def _retrieve(
    con: sqlite3.Connection, run_id: str, query: str, emb: Embedder,
    ids: list[str] | None, M: np.ndarray | None, limit: int,
) -> list[str]:
    """Hybrid keyword+vector retrieval for one query string.

    AND, not OR: this counts how much of the corpus genuinely speaks to a query.
    See db.search_fts.
    """
    kw = db.search_fts(con, run_id, query, limit=limit, op="AND")
    vec: list[str] = []
    if ids is not None and M is not None and len(ids):
        try:
            if emb.name == "tfidf":
                # TF-IDF fits its vocabulary per call, so a separately-encoded
                # query is not comparable to the stored vectors. Skip the vector
                # arm rather than fuse two incompatible spaces; FTS carries it.
                vec = []
            else:
                q = emb.encode([query])[0]
                if M.shape[1] == q.shape[0]:
                    vec = [ids[i] for i in np.argsort(-(M @ q))[:limit]]
        except Exception as e:  # noqa: BLE001
            log.debug("vector arm failed for %r: %s", query, e)
    return rrf(kw, vec) if vec else kw


def attach_coverage(
    clusters: list[DemandCluster],
    run_id: str,
    themes: list[Theme],
    topic: str,
    con: sqlite3.Connection | None = None,
    per_cluster_queries: int = 10,
) -> list[DemandCluster]:
    """For each cluster: how much corpus backs it, and which theme it lands on."""
    own = con is None
    con = con or db.connect()
    try:
        theme_by_doc = _theme_of(themes)
        label_by_theme = {t.id: t.label for t in themes}
        theme_size = {t.id: len(t.doc_ids) for t in themes}
        longform = {
            r["id"] for r in con.execute(
                "SELECT id FROM documents WHERE run_id=? AND platform IN (%s)"
                % ",".join("?" * len(LONGFORM_PLATFORMS)),
                (run_id, *sorted(LONGFORM_PLATFORMS)),
            )
        }
        ids, M = db.load_embeddings(con, run_id)
        emb = get_embedder()

        for c in clusters:
            hits: set[str] = set()
            votes: dict[int, int] = {}
            measured = 0
            # The highest-demand queries carry the cluster's meaning; the long
            # tail adds cost without changing which theme wins.
            for text in c.unique_texts()[:per_cluster_queries]:
                residual = residual_terms(text, topic)
                if not residual:
                    continue  # the bare topic tells us nothing about this cluster
                measured += 1
                for doc_id in _retrieve(
                    con, run_id, residual, emb, ids, M, COVERAGE_LIMIT
                ):
                    if doc_id in longform:
                        continue
                    hits.add(doc_id)
                    tid = theme_by_doc.get(doc_id)
                    if tid is not None:
                        votes[tid] = votes.get(tid, 0) + 1
            c.coverage = min(len(hits), MAX_COVERAGE) if measured else 0
            if votes:
                # Vote *share*, not raw votes. Document length varies enormously
                # across platforms -- a 20,000-character web article matches
                # almost any term set while a 200-character Reddit comment matches
                # few -- so on raw counts the theme holding the longest documents
                # wins every cluster, which is what it did. "What fraction of this
                # theme's documents matched" is the question that actually
                # identifies the theme explaining a demand cluster.
                def share(tid: int) -> float:
                    size = max(theme_size.get(tid, 1), 1)
                    return votes[tid] / size

                best = max(votes, key=share)
                # One document out of three is a coincidence, not an explanation.
                if votes[best] >= MIN_THEME_VOTES:
                    c.theme_id = best
                    c.theme_label = label_by_theme.get(best, "")
        return clusters
    finally:
        if own:
            con.close()


def _percentiles(values: list[float]) -> list[float]:
    """Rank each value in 0..1. Ties share the mean of their positions."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    pct = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        share = (i + j) / 2 / (n - 1)
        for k in range(i, j + 1):
            pct[order[k]] = share
        i = j + 1
    return pct


def compute_gaps(clusters: list[DemandCluster]) -> list[DemandCluster]:
    """gap = demand percentile - coverage percentile, in -1..+1.

    A *relative* measure, deliberately, for the same reason theme momentum is a
    share ratio rather than a raw count: absolute numbers here would mostly track
    how much got harvested and collected, not what is actually under-served.

        gap > 0   people search this more than your corpus discusses it
                  -- demand you have no content for. The actionable output.
        gap ~ 0   demand and conversation are proportionate.
        gap < 0   the conversation is saturated relative to what people search.

    Because both terms are percentiles within this run, a gap is only meaningful
    against the other clusters in the same run.
    """
    if not clusters:
        return clusters
    dem = _percentiles([c.demand_score for c in clusters])
    cov = _percentiles([float(c.coverage) for c in clusters])
    for c, d, v in zip(clusters, dem, cov):
        c.gap = round(d - v, 3)
    clusters.sort(key=lambda c: -c.gap)
    return clusters
