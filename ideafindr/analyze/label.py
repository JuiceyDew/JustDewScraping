"""Name each cluster.

Clustering produces "Theme 0"; a marketing report needs "Chiller noise makes
daily use antisocial". The LLM sees the documents closest to the cluster
centroid, not a random sample, so the label describes the cluster's core.
"""

from __future__ import annotations

import logging

import numpy as np

from ideafindr.analyze.cluster import central_documents
from ideafindr.llm import chat_json_batch
from ideafindr.models import Document, Theme

log = logging.getLogger(__name__)

# Valid stance values, mirroring the Stance literal in models.py. Checked here so
# a hallucinated stance from the batched call falls back to neutral rather than
# failing validation for the whole batch.
STANCES = {"pain_point", "desire", "objection", "praise", "question", "neutral"}

SYSTEM = """You are a market research analyst reading raw social posts.
You name what people are ACTUALLY talking about, in their terms, not in
marketing language. You never invent themes that aren't in the text."""

INSTRUCTIONS = """Each numbered cluster below is a group of posts/comments that were
grouped together because they are about the same thing. Context: the overall topic
is "{topic}", and there are {count} clusters in total.

For EACH cluster, return:
- "coherent": true if its posts really are about one identifiable thing; false if
  it is a grab-bag with no common subject. Be honest -- clustering produces a fixed
  number of groups whether or not that many real themes exist, so a leftovers
  bucket is expected and useful to flag.
- "label": 3-8 words naming the specific thing being discussed. Concrete and
  specific: "Chiller noise makes daily use antisocial", not "Product concerns".
- "description": one sentence on what people are saying and why it matters to a
  marketer.
- "stance": one of pain_point, desire, objection, praise, question, neutral.
  pain_point = something is wrong/broken/frustrating
  desire     = they want something that doesn't exist or they're shopping for it
  objection  = a reason they are NOT buying
  praise     = something works well
  question   = mostly asking how to do something
  neutral    = none of the above"""


def _samples(docs: list[Document], max_chars: int = 320) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        t = " ".join(d.searchable_text.split())[:max_chars]
        out.append(f"{i}. [r/{d.community or '?'}] {t}")
    return "\n".join(out)


def label_themes(
    themes: list[Theme],
    docs: list[Document],
    M: np.ndarray,
    topic: str,
    model: str | None = None,
) -> list[Theme]:
    """Name every theme in a single batched LLM call.

    One call regardless of theme count, rather than one per theme: a 19-theme run
    used to make 19 sequential requests. A failure falls back to keyword labels for
    every theme rather than aborting -- a partially-labelled report still ships.
    """
    by_id = {d.id: d for d in docs}
    index = {d.id: i for i, d in enumerate(docs)}

    blocks: list[str] = []
    targets: list[Theme] = []
    for t in themes:
        central = central_documents(t, by_id, M, index, n=12)
        if not central:
            continue
        blocks.append(_samples(central))
        targets.append(t)

    if not blocks:
        return themes

    try:
        results = chat_json_batch(
            blocks,
            system=SYSTEM,
            instructions=INSTRUCTIONS.format(topic=topic, count=len(blocks)),
            model=model,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("batched theme labelling failed (%s); using keyword labels", e)
        results = [None] * len(blocks)

    for t, res in zip(targets, results):
        if res is None or not res.label.strip():
            t.label = ", ".join(t.keywords[:4]) or f"Theme {t.id}"
            continue
        t.label = res.label.strip()
        t.description = (res.description or "").strip()
        stance = res.stance if res.stance in STANCES else "neutral"
        t.stance = stance  # type: ignore[assignment]
        t.coherent = bool(res.coherent)
    return themes
