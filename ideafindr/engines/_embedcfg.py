"""Shared embedding configuration for the third-party engines.

gpt-researcher and deep-searcher both need an embeddings endpoint of their own --
they build vector stores internally and neither will accept our in-process
TF-IDF embedder. So the engines are only usable when a *real* embeddings API is
reachable, which is exactly what `ideafindr.embed.get_embedder()` resolves.

Failing here with a clear message is much better than letting upstream raise a
confusing 401 from deep inside a vector store.
"""

from __future__ import annotations

from ideafindr.config import settings
from ideafindr.embed import OllamaEmbedder, get_embedder


class EmbeddingsUnavailable(RuntimeError):
    pass


def resolve() -> tuple[str, str]:
    """Return (base_url, api_key) for an OpenAI-compatible embeddings endpoint.

    The base_url keeps its `/v1` suffix -- callers that need the Ollama root
    (langchain_ollama builds its own paths) should use `resolve_root()`.

    Raises EmbeddingsUnavailable when only the TF-IDF fallback is present.
    """
    emb = get_embedder()
    if not isinstance(emb, OllamaEmbedder):
        raise EmbeddingsUnavailable(
            "This command needs an embeddings API, and the stack is configured "
            "cloud-only with no embeddings provider.\n\n"
            "Ollama Cloud does not serve embeddings: /v1/embeddings returns 404 "
            "and /api/embed returns 401 for cloud keys. gpt-researcher and "
            "deep-searcher each build their own vector store and cannot use the "
            "in-process TF-IDF embedder.\n\n"
            "Everything else works without this -- plan, collect, analyze, report "
            "and runs are unaffected.\n\n"
            "To enable these two commands, add an embeddings provider that has an "
            "API (Jina, Voyage, OpenAI, Cohere all work) and point EMBED_BACKEND "
            "at it. See the Embeddings section of the README."
        )
    return emb.base_url, (settings.ollama_api_key if emb.name == "ollama-cloud" else "ollama")


def resolve_root() -> str:
    """The embeddings host WITHOUT the OpenAI `/v1` suffix.

    langchain_ollama's OllamaEmbeddings appends its own `/api/embed` path, so
    handing it a `/v1` URL produces a 404.
    """
    base, _ = resolve()
    return base[: -len("/v1")] if base.endswith("/v1") else base
