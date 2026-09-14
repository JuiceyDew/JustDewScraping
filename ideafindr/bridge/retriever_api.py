"""The bridge: serve our social corpus in gpt-researcher's `custom` retriever format.

gpt-researcher's RETRIEVER=custom expects an endpoint returning:
    [{"url": ..., "raw_content": ...}, ...]

That tiny contract is what lets the entire upstream agent loop -- query planning,
source curation, citation, report writing -- run over scraped Reddit data with no
fork of the upstream repo.

Retrieval is hybrid: SQLite FTS5 keyword matching fused with vector similarity by
reciprocal rank. `raw_content` is the assembled *thread* (post + top comments),
not an isolated comment, because a lone comment gives the agent nothing to judge
or attribute.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np
from fastapi import FastAPI, Query

from ideafindr.config import settings
from ideafindr.embed import Embedder, get_embedder
from ideafindr.models import Document
from ideafindr.store import db

log = logging.getLogger(__name__)

app = FastAPI(title="ideafindr retriever bridge")

_state: dict[str, Any] = {"embedder": None, "vectors": {}}


def _embedder() -> Embedder:
    if _state["embedder"] is None:
        _state["embedder"] = get_embedder()
    return _state["embedder"]


def rrf(*rankings: list[str], k: int = 60) -> list[str]:
    """Reciprocal rank fusion: combine rankings without needing comparable scores.

    Keyword relevance and cosine similarity live on different scales, so fusing
    by rank rather than score avoids inventing a weighting we can't justify.
    """
    scores: dict[str, float] = {}
    for r in rankings:
        for i, doc_id in enumerate(r):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores, key=lambda d: -scores[d])


def _vector_rank(con: sqlite3.Connection, run_id: str, query: str, limit: int) -> list[str]:
    """Rank by cosine similarity, reusing the vectors `analyze` already computed.

    Falls back to embedding the corpus here only when a run was never analysed.
    """
    cache = _state["vectors"]
    if run_id not in cache:
        ids, M = db.load_embeddings(con, run_id)
        if ids is None:
            docs = list(db.iter_documents(con, run_id))
            if not docs:
                return []
            log.warning(
                "run %s has no stored embeddings (was it analysed?); embedding %d "
                "documents now -- this is slow",
                run_id, len(docs),
            )
            ids = [d.id for d in docs]
            M = _embedder().encode([d.searchable_text for d in docs])
        texts = None
        if _embedder().name == "tfidf":
            # TF-IDF fits its vocabulary per call, so stored vectors are not
            # comparable to a separately-encoded query; keep the source text.
            texts = [d.searchable_text for d in db.get_documents(con, run_id, ids)]
        cache[run_id] = (ids, M, texts)
        log.info("bridge: %d vectors ready for run %s", len(ids), run_id)

    ids, M, texts = cache[run_id]
    emb = _embedder()
    if emb.name == "tfidf" and texts:
        M2 = emb.encode([*texts, query])
        q, base = M2[-1], M2[:-1]
    else:
        q, base = emb.encode([query])[0], M
    if base.shape[1] != q.shape[0]:
        log.warning("embedding dim changed (%d stored vs %d now); keyword search only",
                    base.shape[1], q.shape[0])
        return []
    return [ids[i] for i in np.argsort(-(base @ q))[:limit]]


# Embedding cost is driven by text length, not document count: measured against
# a local all-minilm on 4 CPU cores, a batch of 20 short strings took 0.08s per
# document while a batch of 20 x ~2300 characters took 2.47s per document -- 30x.
# gpt-researcher chunks and embeds everything the retriever hands it, so an
# uncapped 9000-character thread is what turns a report into a 20-minute job.
# The agent does not need forty comments; it needs the post and the replies
# people actually upvoted.
THREAD_CHAR_BUDGET = 2600


def assemble_thread(
    con: sqlite3.Connection,
    run_id: str,
    doc: Document,
    max_comments: int = 12,
    char_budget: int | None = THREAD_CHAR_BUDGET,
) -> str:
    """Post + its highest-scoring comments as one readable block.

    Comments are added best-first until the character budget is spent, so what
    survives truncation is what the community endorsed rather than whatever
    happened to be posted earliest.
    """
    parts: list[str] = []
    if doc.title:
        parts.append(f"# {doc.title}")
    parts.append(f"[{doc.platform} · {doc.community or '?'} · {doc.created_at.date()}]")

    body = doc.text.strip()
    if body:
        if char_budget and len(body) > char_budget:
            body = body[:char_budget].rsplit(" ", 1)[0] + " […]"
        parts.append(body)

    if doc.kind == "post":
        kids = db.children_of(con, run_id, doc.id, limit=max_comments)
        if kids:
            used = sum(len(p) for p in parts)
            lines: list[str] = []
            for c in kids:
                score = c.engagement.get("score", 0)
                line = f"({score:+d}) {c.author or 'anon'}: {' '.join(c.text.split())}"
                if char_budget and used + len(line) > char_budget * 2:
                    break
                lines.append(line)
                used += len(line)
            if lines:
                parts.append("\n--- comments ---")
                parts += lines
    return "\n\n".join(parts)


def search_corpus(query: str, run_id: str | None = None, limit: int = 10) -> list[dict[str, str]]:
    """Hybrid keyword+vector retrieval over a run, in gpt-researcher's format.

    Kept separate from the HTTP handler so it can be called (and tested) directly
    -- calling a FastAPI endpoint as a plain function leaks `Query` objects into
    the arguments you don't pass.
    """
    con = db.connect()
    try:
        rid = run_id or db.latest_run_id(con)
        if not rid:
            return []

        kw = db.search_fts(con, rid, query, limit=limit * 4)
        vec = _vector_rank(con, rid, query, limit=limit * 4)
        ranked = rrf(kw, vec)

        docs = {d.id: d for d in db.get_documents(con, rid, ranked[: limit * 4])}

        # Prefer the parent post: it carries the whole discussion with it.
        chosen: list[Document] = []
        used: set[str] = set()
        for did in ranked:
            d = docs.get(did)
            if not d:
                continue
            if d.kind == "comment" and d.parent_id:
                parent = db.get_documents(con, rid, [d.parent_id])
                if parent:
                    d = parent[0]
            if d.id in used:
                continue
            used.add(d.id)
            chosen.append(d)
            if len(chosen) >= limit:
                break

        return [{"url": d.url, "raw_content": assemble_thread(con, rid, d)} for d in chosen]
    finally:
        con.close()


@app.get("/retrieve")
def retrieve(
    query: str = Query(...),
    run_id: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
) -> list[dict[str, str]]:
    return search_corpus(query=query, run_id=run_id, limit=limit)


@app.get("/health")
def health() -> dict[str, Any]:
    con = db.connect()
    try:
        return {"ok": True, "latest_run": db.latest_run_id(con), "embedder": _embedder().name}
    finally:
        con.close()


def serve() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.bridge_host, port=settings.bridge_port, log_level="warning")
