"""Representative verbatims.

Never paraphrased: in a marketing deliverable the customer's own words ARE the
finding. Each quote carries a live permalink so the client can verify it.
"""

from __future__ import annotations

import numpy as np

from ideafindr.analyze.cluster import central_documents
from ideafindr.models import Document, Quote, Theme

# 40 rather than 60: "Chiller noise is unbearable, neighbours complained twice"
# is 55 characters and is exactly the kind of verbatim a report is built on.
# Below ~40 you get fragments ("same here", "this") that carry no information.
MIN_LEN, MAX_LEN = 40, 600


def _clean(text: str) -> str:
    return " ".join(text.split())


def pick_quotes(
    theme: Theme, docs: list[Document], M: np.ndarray, n: int = 4
) -> list[Quote]:
    """Central to the theme, long enough to say something, short enough to read.

    Candidates are ranked centrally-first, then re-ranked by engagement so the
    quotes shown are both on-theme and ones other people actually responded to.
    """
    by_id = {d.id: d for d in docs}
    index = {d.id: i for i, d in enumerate(docs)}
    central = central_documents(theme, by_id, M, index, n=40)

    seen: set[str] = set()
    cands: list[Document] = []
    for d in central:
        t = _clean(d.searchable_text)
        if not (MIN_LEN <= len(t) <= MAX_LEN):
            continue
        fp = t.lower()[:90]
        if fp in seen:
            continue
        seen.add(fp)
        cands.append(d)

    cands.sort(key=lambda d: d.engagement_score(), reverse=True)
    return [
        Quote(
            text=_clean(d.searchable_text)[:MAX_LEN],
            url=d.url,
            author=d.author,
            community=d.community,
            created_at=d.created_at,
            engagement=d.engagement_score(),
        )
        for d in cands[:n]
    ]


def attach_quotes(themes: list[Theme], docs: list[Document], M: np.ndarray, n: int = 4) -> list[Theme]:
    for t in themes:
        t.quotes = pick_quotes(t, docs, M, n=n)
    return themes
