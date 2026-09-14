"""Queries -> intent clusters, scored.

Two jobs: turn raw observations into a defensible demand number, and group the
query space into intents a marketer can act on ("chiller noise and placement"),
rather than handing over 400 loose strings.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np
from pydantic import BaseModel

from ideafindr.analyze.cluster import auto_k, keywords_for
from ideafindr.demand.models import DemandCluster, Query
from ideafindr.embed import Embedder, get_embedder
from ideafindr.llm import chat_json
from ideafindr.models import Stance

log = logging.getLogger(__name__)


def demand_score(observations: list[Query]) -> float:
    """How much demand one query represents. A PROXY, not search volume.

    There is no free source of calibrated volume (Google's official Trends API is
    alpha and application-only as of 2026-09), so this scores the evidence we
    actually have. A query counts for more when:

      - more independent providers suggested it  (agreement across indexes)
      - it ranked higher in their lists          (each provider orders by its own
                                                  popularity model)
      - more different seeds surfaced it         (it sits on many query paths)

    Reciprocal rank is used for the position term, the same choice and for the
    same reason as the bridge's rank fusion: ranks from different providers are
    not comparable as scores, only as orderings.

    The number is meaningful for ranking queries against each other within one
    run. It is not comparable across topics and must never be presented as
    searches per month.
    """
    if not observations:
        return 0.0
    sources = {o.source for o in observations}
    seeds = {o.seed for o in observations}
    best_rank = min(o.rank for o in observations)
    return round(len(sources) * (1.0 / (1 + best_rank)) * (1 + math.log(len(seeds))), 4)


def score_queries(queries: list[Query]) -> dict[str, float]:
    """Score every distinct query in a harvest."""
    grouped: dict[str, list[Query]] = {}
    for q in queries:
        grouped.setdefault(q.key, []).append(q)
    return {k: demand_score(v) for k, v in grouped.items()}


SYSTEM = """You are a marketing analyst reading real search queries people typed
into Google, YouTube, Bing and DuckDuckGo about one topic. You name the INTENT
behind a group of queries -- what the searcher is trying to find out or decide --
in their terms, never in marketing language."""

PROMPT = """These search queries were grouped together because they express the
same intent. Topic context: {topic}

{samples}

Return JSON:
- "coherent": true if these really do share one intent; false if the grouping is
  a grab-bag. Clustering returns a fixed number of groups whether or not that many
  real intents exist, so a leftovers bucket is expected and useful to flag.
- "label": 3-8 words naming the intent. Concrete: "Sizing a chiller for a DIY
  build", not "Product research".
- "description": one sentence on what the searcher wants and why it matters to a
  marketer.
- "stance": one of pain_point, desire, objection, praise, question, neutral.
  pain_point = something is wrong/broken/frustrating
  desire     = they want something or are shopping for it
  objection  = a reason they might NOT buy
  question   = mostly asking how something works
  neutral    = none of the above"""


class _Label(BaseModel):
    label: str
    description: str = ""
    stance: Stance = "neutral"
    coherent: bool = True


def cluster_queries(
    queries: list[Query],
    embedder: Embedder | None = None,
    n_clusters: int | None = None,
) -> list[DemandCluster]:
    """Group distinct queries by intent.

    Note the embedder matters more here than it does for documents. TF-IDF groups
    by shared words, and search queries share almost nothing -- "chiller too loud"
    and "compressor noise at night" have no token in common. With TF-IDF the
    clusters come out lexical; with a real embedder they come out semantic. See
    EMBED_BASE_URL in .env.example.
    """
    from sklearn.cluster import KMeans

    if not queries:
        return []

    scores = score_queries(queries)
    by_key: dict[str, list[Query]] = {}
    for q in queries:
        by_key.setdefault(q.key, []).append(q)

    keys = sorted(by_key, key=lambda k: -scores[k])
    if len(keys) < 6:  # too few to partition meaningfully
        obs = [o for k in keys for o in by_key[k]]
        return [
            DemandCluster(
                id=0, label="All queries", queries=obs,
                demand_score=round(sum(scores[k] for k in keys), 3),
                sources=dict(Counter(o.source for o in obs)),
                keywords=keywords_for(keys),
            )
        ]

    emb = embedder or get_embedder()
    M = emb.encode(keys)
    k = min(n_clusters or auto_k(len(keys)), len(keys))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(M)
    log.info("clustered %d distinct queries into %d intents (%s)", len(keys), k, emb.name)

    groups: dict[int, list[str]] = {}
    for key, lab in zip(keys, labels):
        groups.setdefault(int(lab), []).append(key)

    clusters: list[DemandCluster] = []
    for cid, ks in enumerate(
        sorted(groups.values(), key=lambda g: -sum(scores[x] for x in g))
    ):
        obs = [o for k_ in ks for o in by_key[k_]]
        clusters.append(
            DemandCluster(
                id=cid,
                label=", ".join(ks[:3]),  # placeholder until the LLM names it
                queries=obs,
                demand_score=round(sum(scores[x] for x in ks), 3),
                sources=dict(Counter(o.source for o in obs)),
                keywords=keywords_for(ks),
            )
        )
    return clusters


def label_clusters(
    clusters: list[DemandCluster], topic: str, model: str | None = None
) -> list[DemandCluster]:
    """Name each intent. A failure on one cluster keeps its keyword label."""
    for c in clusters:
        sample = c.unique_texts()[:20]
        if not sample:
            continue
        try:
            res = chat_json(
                PROMPT.format(
                    topic=topic,
                    samples="\n".join(f"- {s}" for s in sample),
                ),
                _Label,
                system=SYSTEM,
                model=model,
            )
            c.label = res.label.strip() or c.label
            c.description = res.description.strip()
        except Exception as e:  # noqa: BLE001
            log.warning("labelling demand cluster %d failed: %s", c.id, e)
            c.label = ", ".join(c.keywords[:4]) or c.label
    return clusters
