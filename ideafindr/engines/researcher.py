"""gpt-researcher, pointed at our own corpus.

Used as a library with no fork: upstream's RETRIEVER=custom reads an endpoint
that returns [{"url", "raw_content"}], which is exactly what the bridge serves.
Setting RETRIEVER="custom,duckduckgo" makes one report draw on both the scraped
social corpus and the live web.

gpt-researcher is a fast-moving dependency and its env var names have changed
between releases -- if a report comes back citing only web sources, check that
`RETRIEVER_ENDPOINT` is still the name upstream reads.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

import httpx

from ideafindr.config import settings

log = logging.getLogger(__name__)

# Generous on purpose: a slow local embed is better than a failed report.
EMBED_HTTP_TIMEOUT = 900


class BridgeServer:
    """Runs the retriever bridge in a background thread for the duration of a report."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._server = None

    def start(self, timeout: float = 25.0) -> None:
        import uvicorn

        from ideafindr.bridge.retriever_api import app

        cfg = uvicorn.Config(
            app, host=settings.bridge_host, port=settings.bridge_port, log_level="warning"
        )
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if httpx.get(f"{settings.bridge_url}/health", timeout=2).status_code == 200:
                    log.info("bridge up at %s", settings.bridge_url)
                    return
            except Exception:  # noqa: BLE001
                time.sleep(0.25)
        raise RuntimeError(f"bridge failed to start on {settings.bridge_url}")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=10)

    def __enter__(self) -> "BridgeServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def write_report(
    topic: str,
    run_id: str,
    report_type: str = "research_report",
    include_web: bool = True,
) -> str:
    """Run gpt-researcher over our corpus (+ the live web) and return the report."""
    try:
        from gpt_researcher import GPTResearcher
    except ImportError as e:
        raise RuntimeError(
            "gpt-researcher is not installed. Install the engines extra:\n"
            "  uv sync --extra engines"
        ) from e

    # gpt-researcher builds its own vector store, so it needs a reachable
    # embeddings API -- our in-process TF-IDF embedder cannot serve it. Resolve
    # (and fail clearly) before starting the bridge or burning any tokens.
    from ideafindr.engines._embedcfg import resolve_root

    embed_root = resolve_root()

    settings.export_openai_env()
    retrievers = "custom,duckduckgo" if include_web else "custom"
    os.environ["RETRIEVER"] = retrievers
    os.environ["RETRIEVER_ENDPOINT"] = f"{settings.bridge_url}/retrieve"
    # Upstream forwards RETRIEVER_ARG_* to the endpoint as query params, which is
    # how the bridge learns which run's corpus to search.
    os.environ["RETRIEVER_ARG_RUN_ID"] = run_id
    # Chat and embeddings go to DIFFERENT hosts: chat to Ollama Cloud, embeddings
    # to the local daemon (Cloud serves no embeddings). gpt-researcher's `ollama`
    # embedding provider reads OLLAMA_BASE_URL, which is separate from the
    # OPENAI_BASE_URL that export_openai_env() points at Ollama Cloud.
    #
    # Do NOT set OPENAI_BASE_URL to the embeddings host here: it silently
    # redirects the chat calls too, and every LLM request then 404s with
    # "model 'gpt-oss:120b' not found" because the local daemon only has the
    # embedding model.
    os.environ["EMBEDDING"] = f"ollama:{settings.embed_model}"
    os.environ["OLLAMA_BASE_URL"] = embed_root

    async def _run() -> str:
        r = GPTResearcher(query=topic, report_type=report_type)

        # Raise the embeddings HTTP timeout. A CPU-only Ollama takes tens of
        # seconds per batch (embedding cost scales with text length, and the
        # retriever hands over whole threads), which blows past the ollama
        # client's default and surfaces as httpx.ReadTimeout -> the report comes
        # back None -> "object of type 'NoneType' has no len()".
        #
        # gpt-researcher never populates Config.embedding_kwargs from the
        # environment, so there is no config path for this; rebuilding the public
        # `memory` attribute is the only seam. Re-check on upgrade.
        try:
            from gpt_researcher.memory import Memory

            r.memory = Memory(
                r.cfg.embedding_provider,
                r.cfg.embedding_model,
                client_kwargs={"timeout": EMBED_HTTP_TIMEOUT},
            )
            log.info("embeddings timeout raised to %ss", EMBED_HTTP_TIMEOUT)
        except Exception as e:  # noqa: BLE001
            log.warning("could not raise the embeddings timeout: %s", e)

        await r.conduct_research()
        return await r.write_report()

    with BridgeServer():
        return asyncio.run(_run())
