"""Name each cluster.

Clustering produces "Theme 0"; a marketing report needs "Chiller noise makes
daily use antisocial". The LLM sees the documents closest to the cluster
centroid, not a random sample, so the label describes the cluster's core.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, Field

from ideafindr.analyze.cluster import central_documents
from ideafindr.llm import chat_json
from ideafindr.models import Document, Stance, Theme

log = logging.getLogger(__name__)

SYSTEM = """You are a market research analyst reading raw social posts.
You name what people are ACTUALLY talking about, in their terms, not in
marketing language. You never invent themes that aren't in the text."""

PROMPT = """These posts/comments were grouped together because they are about the
same thing. Topic context: {topic}

{samples}

Return JSON:
- "coherent": true if these posts really are about one identifiable thing;
  false if they are a grab-bag with no common subject. Be honest -- clustering
  produces a fixed number of groups whether or not that many real themes exist,
  so a leftovers bucket is expected and useful to flag.
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


class _Label(BaseModel):
    label: str
    description: str = ""
    stance: Stance = "neutral"
    coherent: bool = True


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
    """Label in place. A failure on one theme falls back to its keywords rather
    than aborting the run -- a partially-labelled report still ships."""
    by_id = {d.id: d for d in docs}
    index = {d.id: i for i, d in enumerate(docs)}

    for t in themes:
        central = central_documents(t, by_id, M, index, n=12)
        if not central:
            continue
        try:
            res = chat_json(
                PROMPT.format(topic=topic, samples=_samples(central)),
                _Label,
                system=SYSTEM,
                model=model,
            )
            t.label = res.label.strip() or t.label
            t.description = res.description.strip()
            t.stance = res.stance
            t.coherent = res.coherent
        except Exception as e:  # noqa: BLE001
            log.warning("labelling theme %d failed: %s", t.id, e)
            t.label = ", ".join(t.keywords[:4]) or f"Theme {t.id}"
    return themes
