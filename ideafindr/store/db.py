"""SQLite store: raw documents, runs, themes.

FTS5 gives the bridge a keyword arm to fuse with vector similarity, and costs
nothing since SQLite ships with it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from ideafindr.config import settings
from ideafindr.models import Document, Run, RunPlan, Theme

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    topic       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    plan        TEXT NOT NULL,
    sources     TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    platform    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    url         TEXT NOT NULL,
    title       TEXT,
    text        TEXT NOT NULL,
    author      TEXT,
    created_at  TEXT NOT NULL,
    parent_id   TEXT,
    community   TEXT,
    engagement  TEXT NOT NULL DEFAULT '{}',
    raw         TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, id)
);
CREATE INDEX IF NOT EXISTS ix_docs_run      ON documents(run_id);
CREATE INDEX IF NOT EXISTS ix_docs_parent   ON documents(run_id, parent_id);
CREATE INDEX IF NOT EXISTS ix_docs_created  ON documents(run_id, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    doc_id UNINDEXED, run_id UNINDEXED, title, text, tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS themes (
    run_id      TEXT NOT NULL,
    id          INTEGER NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (run_id, id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    run_id      TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL,
    PRIMARY KEY (run_id, doc_id)
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or settings.db_path
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


# --- runs ---------------------------------------------------------------------


def create_run(con: sqlite3.Connection, run_id: str, plan: RunPlan, sources: list[str]) -> Run:
    now = datetime.now(timezone.utc)
    con.execute(
        "INSERT OR REPLACE INTO runs (id, topic, created_at, plan, sources) VALUES (?,?,?,?,?)",
        (run_id, plan.topic, now.isoformat(), plan.model_dump_json(), json.dumps(sources)),
    )
    con.commit()
    return Run(id=run_id, topic=plan.topic, created_at=now, plan=plan, sources=sources)


def get_run(con: sqlite3.Connection, run_id: str) -> Run | None:
    r = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not r:
        return None
    n = con.execute("SELECT COUNT(*) c FROM documents WHERE run_id=?", (run_id,)).fetchone()["c"]
    return Run(
        id=r["id"],
        topic=r["topic"],
        created_at=datetime.fromisoformat(r["created_at"]),
        plan=RunPlan.model_validate_json(r["plan"]),
        sources=json.loads(r["sources"]),
        doc_count=n,
    )


def list_runs(con: sqlite3.Connection) -> list[Run]:
    ids = [r["id"] for r in con.execute("SELECT id FROM runs ORDER BY created_at DESC")]
    return [r for r in (get_run(con, i) for i in ids) if r]


def latest_run_id(con: sqlite3.Connection) -> str | None:
    r = con.execute("SELECT id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
    return r["id"] if r else None


# --- documents ----------------------------------------------------------------


def save_documents(con: sqlite3.Connection, run_id: str, docs: Iterable[Document]) -> int:
    """Insert, skipping documents already stored for this run. Returns new count."""
    n = 0
    for d in docs:
        cur = con.execute(
            """INSERT OR IGNORE INTO documents
               (id, run_id, platform, kind, url, title, text, author, created_at,
                parent_id, community, engagement, raw)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d.id, run_id, d.platform, d.kind, d.url, d.title, d.text, d.author,
                d.created_at.isoformat(), d.parent_id, d.community,
                json.dumps(d.engagement), json.dumps(d.raw, default=str),
            ),
        )
        if cur.rowcount:
            con.execute(
                "INSERT INTO documents_fts (doc_id, run_id, title, text) VALUES (?,?,?,?)",
                (d.id, run_id, d.title or "", d.text),
            )
            n += 1
    con.commit()
    return n


def _row_to_doc(r: sqlite3.Row) -> Document:
    return Document(
        id=r["id"], platform=r["platform"], kind=r["kind"], url=r["url"],
        title=r["title"], text=r["text"], author=r["author"],
        created_at=datetime.fromisoformat(r["created_at"]),
        parent_id=r["parent_id"], community=r["community"],
        engagement=json.loads(r["engagement"]), raw=json.loads(r["raw"]),
    )


def iter_documents(
    con: sqlite3.Connection, run_id: str, kinds: list[str] | None = None
) -> Iterator[Document]:
    q = "SELECT * FROM documents WHERE run_id=?"
    args: list = [run_id]
    if kinds:
        q += f" AND kind IN ({','.join('?' * len(kinds))})"
        args += kinds
    q += " ORDER BY created_at"
    for r in con.execute(q, args):
        yield _row_to_doc(r)


def get_documents(con: sqlite3.Connection, run_id: str, doc_ids: list[str]) -> list[Document]:
    if not doc_ids:
        return []
    out: list[Document] = []
    for i in range(0, len(doc_ids), 500):  # SQLite variable limit
        chunk = doc_ids[i : i + 500]
        q = f"SELECT * FROM documents WHERE run_id=? AND id IN ({','.join('?' * len(chunk))})"
        out += [_row_to_doc(r) for r in con.execute(q, [run_id, *chunk])]
    return out


def search_fts(con: sqlite3.Connection, run_id: str, query: str, limit: int = 50) -> list[str]:
    """Keyword arm of the bridge's hybrid retrieval. Returns doc_ids, best first."""
    # Strip FTS5 operators; user/LLM queries are natural language, not FTS syntax.
    safe = "".join(c if c.isalnum() or c.isspace() else " " for c in query).strip()
    if not safe:
        return []
    terms = " OR ".join(safe.split())
    try:
        rows = con.execute(
            """SELECT doc_id FROM documents_fts
               WHERE documents_fts MATCH ? AND run_id=?
               ORDER BY rank LIMIT ?""",
            (terms, run_id, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r["doc_id"] for r in rows]


def children_of(con: sqlite3.Connection, run_id: str, parent_id: str, limit: int = 25) -> list[Document]:
    """Comments under a post -- the bridge assembles these into a thread."""
    rows = con.execute(
        "SELECT * FROM documents WHERE run_id=? AND parent_id=? LIMIT ?",
        (run_id, parent_id, limit),
    ).fetchall()
    docs = [_row_to_doc(r) for r in rows]
    docs.sort(key=lambda d: d.engagement_score(), reverse=True)
    return docs


# --- embeddings ---------------------------------------------------------------


def save_embeddings(
    con: sqlite3.Connection, run_id: str, doc_ids: list[str], matrix
) -> None:
    """Persist the analysis embeddings so the bridge doesn't recompute them.

    Without this the retriever re-embeds the whole corpus on its first query --
    hundreds of documents through a CPU model, several minutes, for vectors that
    `analyze` already produced moments earlier.
    """
    import numpy as np

    m = np.asarray(matrix, dtype=np.float32)
    con.execute("DELETE FROM embeddings WHERE run_id=?", (run_id,))
    con.executemany(
        "INSERT INTO embeddings (run_id, doc_id, dim, vec) VALUES (?,?,?,?)",
        [(run_id, d, int(m.shape[1]), m[i].tobytes()) for i, d in enumerate(doc_ids)],
    )
    con.commit()


def load_embeddings(con: sqlite3.Connection, run_id: str):
    """Returns (doc_ids, matrix) or (None, None) when nothing is stored."""
    import numpy as np

    rows = con.execute(
        "SELECT doc_id, dim, vec FROM embeddings WHERE run_id=?", (run_id,)
    ).fetchall()
    if not rows:
        return None, None
    dim = rows[0]["dim"]
    ids = [r["doc_id"] for r in rows]
    m = np.vstack([np.frombuffer(r["vec"], dtype=np.float32, count=dim) for r in rows])
    return ids, m


# --- themes -------------------------------------------------------------------


def save_themes(con: sqlite3.Connection, run_id: str, themes: list[Theme]) -> None:
    con.execute("DELETE FROM themes WHERE run_id=?", (run_id,))
    con.executemany(
        "INSERT INTO themes (run_id, id, payload) VALUES (?,?,?)",
        [(run_id, t.id, t.model_dump_json()) for t in themes],
    )
    con.commit()


def load_themes(con: sqlite3.Connection, run_id: str) -> list[Theme]:
    rows = con.execute("SELECT payload FROM themes WHERE run_id=? ORDER BY id", (run_id,))
    return [Theme.model_validate_json(r["payload"]) for r in rows]
