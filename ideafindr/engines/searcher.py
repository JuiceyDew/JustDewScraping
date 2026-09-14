"""deep-searcher: follow-up Q&A over a corpus that has already been collected.

Lets an analyst interrogate a run without re-scraping. Upstream warns that small
models fail to follow its prompts, so this uses SMART_MODEL rather than the fast
one.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ideafindr.config import settings
from ideafindr.store import db

log = logging.getLogger(__name__)


def export_corpus(run_id: str, out_dir: Path | None = None) -> Path:
    """Write the corpus as markdown files for deep-searcher to ingest.

    One file per thread rather than one per document: deep-searcher chunks what
    it is given, and a chunk that is a lone comment has no context to answer with.
    """
    from ideafindr.bridge.retriever_api import assemble_thread

    out = out_dir or (settings.reports_dir / run_id / "corpus")
    out.mkdir(parents=True, exist_ok=True)

    con = db.connect()
    try:
        posts = [d for d in db.iter_documents(con, run_id) if d.kind != "comment"]
        for d in posts:
            # No char budget here: this is a one-time offline ingest, not a
            # per-query embed on the critical path.
            body = assemble_thread(con, run_id, d, max_comments=40, char_budget=None)
            safe = d.id.replace(":", "_").replace("/", "_")
            (out / f"{safe}.md").write_text(f"Source: {d.url}\n\n{body}", encoding="utf-8")
        log.info("exported %d threads to %s", len(posts), out)
        return out
    finally:
        con.close()


def _configure() -> None:
    from deepsearcher.configuration import Configuration, init_config

    from ideafindr.engines._embedcfg import resolve

    embed_base, embed_key = resolve()

    cfg = Configuration()
    cfg.set_provider_config(
        "llm",
        "OpenAI",
        {
            "model": settings.smart_model,
            "base_url": settings.ollama_base_url,
            "api_key": settings.ollama_api_key,
        },
    )
    # Whichever embeddings endpoint actually answered -- cloud or local ollama.
    cfg.set_provider_config("embedding", "OpenAIEmbedding", {
        "model": settings.embed_model,
        "base_url": embed_base,
        "api_key": embed_key,
    })
    cfg.set_provider_config(
        "vector_db", "Milvus", {"uri": str(settings.db_path.parent / "milvus.db")}
    )
    init_config(config=cfg)


def ask(run_id: str, question: str) -> str:
    try:
        from deepsearcher.offline_loading import load_from_local_files
        from deepsearcher.online_query import query
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("deepsearcher"):
            raise RuntimeError(
                "deepsearcher is not installed. Install the engines extra:\n"
                "  uv sync --extra engines"
            ) from e
        raise RuntimeError(
            f"deepsearcher is installed but one of its dependencies is missing: {e.name}.\n"
            "Re-sync the engines extra:  uv sync --extra engines"
        ) from e
    except ImportError as e:
        # deepsearcher IS installed but fails to import -- almost always a
        # transitive dependency that changed its API underneath it. Saying "not
        # installed" here sends people to reinstall something that is already
        # there; name the real broken import instead.
        raise RuntimeError(
            f"deepsearcher is installed but cannot be imported: {e}\n"
            "This is an upstream dependency conflict, not a missing package. "
            "Known case: firecrawl-py >=3 renamed ScrapeOptions, which deepsearcher "
            "still imports -- pyproject pins firecrawl-py<3 for exactly this.\n"
            "Try:  uv sync --extra engines --reinstall-package firecrawl-py"
        ) from e

    _configure()
    corpus = export_corpus(run_id)
    load_from_local_files(paths_or_directory=str(corpus), collection_name=f"run_{run_id.replace('-','_')}")
    answer, *_ = query(question)
    return answer
