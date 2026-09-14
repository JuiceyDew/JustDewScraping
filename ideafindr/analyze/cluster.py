"""Documents -> themes.

Spherical k-means: KMeans over L2-normalised vectors, where euclidean distance is
monotonic with cosine. Hierarchical linkage was tried first and abandoned -- see
the measurements in cluster_documents(): on an 805-document corpus it kept
peeling small groups off one dense core, leaving a single "theme" holding 43-51%
of the corpus, which is not a finding anyone can act on. k-means balances by
construction and is faster on big corpora.

k is not knowable in advance, so auto_k() derives a readable target from corpus
size and _split_oversized() breaks up any cluster that still swallows too much.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import numpy as np

from ideafindr.analyze.signals import strip_noise
from ideafindr.embed import Embedder, get_embedder
from ideafindr.models import Document, Theme

log = logging.getLogger(__name__)

MIN_CONTENT_CHARS = 15

_WORD = re.compile(r"[a-z][a-z'-]{2,}")
_STOP = set(
    """the a an and or but if then than that this these those it its is are was were be been
    being have has had do does did will would can could should i you he she they we me my your
    of to in for on at by with from as so not no yes just like get got really very much more
    most some any all what when where who how why there here about into out up down over
    only also even still back way too lot thing things know think want need make made going
    im ive dont doesnt didnt cant wont thats youre theyre isnt arent hes shes well good bad
    time people day days one two first new old right now""".split()
)

MIN_THEME_SIZE = 3


def keywords_for(texts: list[str], n: int = 8) -> list[str]:
    """Distinctive-ish words for a cluster. Cheap; the LLM does the real naming."""
    c: Counter[str] = Counter()
    for t in texts:
        c.update({w for w in _WORD.findall(strip_noise(t).lower()) if w not in _STOP})
    return [w for w, _ in c.most_common(n)]


def auto_k(n: int) -> int:
    """Target cluster count for a corpus of n documents.

    Chosen over a fixed cosine distance threshold because the right threshold
    depends entirely on the embedder: TF-IDF vectors put paraphrases ~0.9 apart
    while a neural embedder puts them ~0.3 apart, so any hardcoded threshold
    either shatters the corpus or collapses it. Cluster *count* is stable across
    both, and a marketing report wants a readable number of themes anyway.
    """
    return int(max(4, min(28, round((n / 2) ** 0.5))))


def _split_oversized(
    groups: dict[int, list[int]],
    M: np.ndarray,
    max_share: float = 0.25,
    min_size: int = MIN_THEME_SIZE,
    depth: int = 3,
) -> dict[int, list[int]]:
    """Recursively split any cluster holding more than `max_share` of the corpus.

    Ward gives more balanced clusters than average linkage, but on a corpus where
    most documents are genuinely about one subject it still yields a single blob
    holding half the documents. "Half your corpus is about the topic" is not a
    finding -- a report needs that blob broken into the distinct things people
    say *within* it. Splitting only the oversized clusters leaves the already
    well-separated ones untouched.
    """
    from sklearn.cluster import KMeans

    total = sum(len(g) for g in groups.values())
    if total == 0:
        return groups

    out: dict[int, list[int]] = {}
    next_id = 0
    queue = [(g, depth) for g in groups.values()]

    while queue:
        idxs, d = queue.pop()
        too_big = len(idxs) / total > max_share
        if not too_big or d <= 0 or len(idxs) < min_size * 2:
            out[next_id] = idxs
            next_id += 1
            continue
        sub = M[idxs]
        parts = min(4, max(2, round(len(idxs) / (max_share * total))))
        try:
            lab = KMeans(
                n_clusters=min(parts, len(idxs)), n_init=10, random_state=0
            ).fit_predict(sub)
        except ValueError:
            out[next_id] = idxs
            next_id += 1
            continue
        split: dict[int, list[int]] = {}
        for pos, l in enumerate(lab):
            split.setdefault(int(l), []).append(idxs[pos])
        if len(split) < 2:
            out[next_id] = idxs
            next_id += 1
            continue
        log.info("split a %d-doc cluster into %d", len(idxs), len(split))
        queue += [(g, d - 1) for g in split.values()]

    return out


def cluster_documents(
    docs: list[Document],
    embedder: Embedder | None = None,
    n_clusters: int | None = None,
    min_size: int = MIN_THEME_SIZE,
    max_share: float = 0.25,
) -> tuple[list[Theme], np.ndarray, list[Document]]:
    """Returns (themes, embedding matrix, the documents the matrix is aligned to).

    The third element matters: content-free documents are filtered out here, so
    row i of M is `kept[i]`, NOT `docs[i]`. Callers that index into M (quote
    selection, centroid ranking) must use the returned list or they will silently
    mis-attribute quotes.

    Clusters smaller than `min_size` are dropped as noise rather than forced into
    a neighbour -- a 'theme' of one post is an anecdote, not a theme.
    """
    from sklearn.cluster import KMeans

    if not docs:
        return [], np.zeros((0, 0), dtype=np.float32), []

    # Drop content-free documents ("this", "lol", "[deleted]") before clustering:
    # they carry no signal and only dilute every theme they land in.
    docs = [d for d in docs if len(d.searchable_text.strip()) >= MIN_CONTENT_CHARS]
    if not docs:
        return [], np.zeros((0, 0), dtype=np.float32), []

    emb = embedder or get_embedder()
    texts = [d.searchable_text for d in docs]
    M = emb.encode(texts)
    log.info("embedded %d documents (%s, dim=%d)", len(docs), emb.name, M.shape[1])

    if len(docs) < min_size * 2:
        t = Theme(id=0, label="All discussion", doc_ids=[d.id for d in docs],
                  keywords=keywords_for(texts))
        return [t], M, docs

    k = min(n_clusters or auto_k(len(docs)), len(docs))
    # Spherical k-means: KMeans on L2-normalised vectors, where euclidean distance
    # is monotonic with cosine. Measured on an 805-document cold-plunge corpus,
    # largest-cluster share was: average-linkage cosine ~43%, Ward 51%, k-means
    # 20%. Hierarchical linkage keeps peeling small groups off one dense core,
    # which yields a "theme" holding half the corpus -- not a finding anyone can
    # act on. k-means balances by construction and is faster on big corpora.
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(M)

    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)

    # Only auto-balance when the caller did not pick k: `--clusters 3` is an
    # explicit request for three themes, and silently returning six is a worse
    # answer than an unbalanced three.
    if n_clusters is None:
        groups = _split_oversized(groups, M, max_share=max_share, min_size=min_size)

    kept = sorted(
        (g for g in groups.values() if len(g) >= min_size), key=len, reverse=True
    )
    noise = sum(len(g) for g in groups.values() if len(g) < min_size)
    log.info(
        "clustered into %d themes (%d docs), %d noise docs dropped (%.0f%%)",
        len(kept), sum(len(g) for g in kept), noise, 100 * noise / max(1, len(docs)),
    )

    themes: list[Theme] = []
    for tid, idxs in enumerate(kept):
        themes.append(
            Theme(
                id=tid,
                label=f"Theme {tid}",
                doc_ids=[docs[i].id for i in idxs],
                keywords=keywords_for([texts[i] for i in idxs]),
            )
        )
    return themes, M, docs


def central_documents(
    theme: Theme, docs_by_id: dict[str, Document], M: np.ndarray, index: dict[str, int], n: int = 12
) -> list[Document]:
    """The documents closest to the cluster centroid -- what the theme is 'about'.

    Used to give the labelling LLM a representative sample instead of a random one.
    """
    idxs = [index[d] for d in theme.doc_ids if d in index]
    if not idxs:
        return []
    sub = M[idxs]
    centroid = sub.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    order = np.argsort(-(sub @ centroid))
    return [docs_by_id[theme.doc_ids[i]] for i in order[:n] if theme.doc_ids[i] in docs_by_id]
