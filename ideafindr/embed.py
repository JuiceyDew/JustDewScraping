"""Embedding backends, chosen at runtime.

Embeddings are remote or in-process, never a local model: this deployment runs
on low-end hardware, so nothing may run inference on this box. The chain `auto`
tries, in order:

    api           -> any OpenAI-compatible /v1/embeddings provider (Jina, Voyage,
                     OpenAI, Cohere, Together). Set EMBED_BASE_URL + EMBED_API_KEY.
    ollama-cloud  -> Ollama Cloud, if it ever starts serving embeddings. Verified
                     2026-09: /v1/embeddings returns 404 and /api/embed returns
                     401 for cloud keys, so this probe fails and `auto` moves on.
    tfidf         -> pure scikit-learn, in-process, no API, no network.

`ollama-local` exists too but is deliberately NOT in the `auto` chain -- if a
daemon ever appears on this machine for unrelated reasons, `auto` must not start
silently routing embedding work onto the CPU. Ask for it by name
(EMBED_BACKEND=ollama-local) on hardware that can afford it.

TF-IDF+SVD is a real fallback, not a stub: for short social posts that share
vocabulary it clusters decently. It is weaker at grouping posts that mean the
same thing in different words, which is exactly what the LLM labelling step
partially compensates for. It is weaker still on short search queries, which
share almost no vocabulary at all -- so the demand layer wants a real embedder.
"""

from __future__ import annotations

import logging

import httpx
import numpy as np

from ideafindr.config import settings

log = logging.getLogger(__name__)


class Embedder:
    """Base: turn texts into a (n, dim) L2-normalised float32 matrix."""

    name = "base"
    dim = 0

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _l2(m: np.ndarray) -> np.ndarray:
    """L2-normalise, replacing zero rows with distinct pseudo-random unit vectors.

    A document whose text is entirely stopwords yields an all-zero TF-IDF row, and
    sklearn's cosine linkage raises on zero vectors. Giving each one a *distinct*
    random direction (rather than a shared constant) keeps them near-orthogonal to
    everything, so they scatter into singleton clusters and get dropped as noise --
    which is the correct fate for a content-free document.
    """
    if m.size == 0:
        return m.astype(np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    dead = (n < 1e-9).ravel()
    if dead.any():
        rng = np.random.default_rng(0)
        m = m.copy()
        m[dead] = rng.standard_normal((int(dead.sum()), m.shape[1]))
        n = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.maximum(n, 1e-12)).astype(np.float32)


class OpenAICompatEmbedder(Embedder):
    """Any OpenAI-compatible /v1/embeddings endpoint.

    The protocol is identical across hosted providers (Jina, Voyage, OpenAI,
    Cohere, Together) and an Ollama daemon's OpenAI shim: POST {base}/embeddings
    with {"model", "input"} and a Bearer token, then read data[].embedding back in
    index order. One class covers all of them, so switching provider is a config
    change rather than a code change.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "", name: str = "api"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name
        # Kept on the instance so callers (engines/_embedcfg.py) can hand the
        # credential to a third-party library without re-deriving it from `name`.
        self.api_key = api_key
        self._h = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def encode(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out: list[list[float]] = []
        with httpx.Client(timeout=120) as c:
            for i in range(0, len(texts), batch):
                chunk = [t[:6000] or " " for t in texts[i : i + batch]]
                r = c.post(
                    f"{self.base_url}/embeddings",
                    headers=self._h,
                    json={"model": self.model, "input": chunk},
                )
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
                out += [d["embedding"] for d in data]
        m = np.asarray(out, dtype=np.float32)
        self.dim = m.shape[1] if m.size else 0
        return _l2(m)


class TfidfEmbedder(Embedder):
    """TF-IDF -> TruncatedSVD (LSA). Pure sklearn: in-process, offline, free."""

    name = "tfidf"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        safe = [t if t.strip() else "empty" for t in texts]
        vec = TfidfVectorizer(
            max_features=40_000,
            ngram_range=(1, 2),
            min_df=2 if len(safe) > 40 else 1,
            max_df=0.6,
            stop_words="english",
            sublinear_tf=True,
        )
        try:
            X = vec.fit_transform(safe)
        except ValueError:  # every term pruned (tiny or degenerate corpus)
            vec = TfidfVectorizer(stop_words="english")
            X = vec.fit_transform(safe)

        k = int(min(self.dim, X.shape[1] - 1, max(2, len(safe) - 1)))
        if k < 2:
            return _l2(np.asarray(X.todense(), dtype=np.float32))
        svd = TruncatedSVD(n_components=k, random_state=0)
        m = svd.fit_transform(X).astype(np.float32)
        self.dim = m.shape[1]
        return _l2(m)


def _probe(base: str, model: str, key: str = "") -> bool:
    try:
        h = {"Authorization": f"Bearer {key}"} if key else {}
        r = httpx.post(
            f"{base.rstrip('/')}/embeddings",
            headers=h,
            json={"model": model, "input": "probe"},
            timeout=25,
        )
        return r.status_code == 200 and bool(r.json().get("data"))
    except Exception:  # noqa: BLE001
        return False


def get_embedder(backend: str | None = None) -> Embedder:
    """Resolve the backend. `auto` probes api -> ollama-cloud -> tfidf.

    `ollama-local` is reachable only by naming it explicitly; see the module
    docstring for why it is kept out of `auto`.
    """
    b = (backend or settings.embed_backend).lower()

    if b in ("api", "auto"):
        if not settings.embed_base_url:
            if b == "api":
                raise RuntimeError(
                    "EMBED_BACKEND=api but EMBED_BASE_URL is empty. Point it at an "
                    "OpenAI-compatible embeddings endpoint (Jina, Voyage, OpenAI, "
                    "Cohere and Together all work) and set EMBED_API_KEY. "
                    "See .env.example."
                )
        elif _probe(settings.embed_base_url, settings.embed_model, settings.embed_api_key):
            log.info("embeddings: api (%s @ %s)", settings.embed_model, settings.embed_base_url)
            return OpenAICompatEmbedder(
                settings.embed_base_url, settings.embed_model,
                settings.embed_api_key, "api",
            )
        elif b == "api":
            raise RuntimeError(
                f"EMBED_BACKEND=api but {settings.embed_base_url}/embeddings did not "
                f"return usable embeddings for model {settings.embed_model!r}. "
                "Check EMBED_BASE_URL, EMBED_API_KEY and EMBED_MODEL, then run: "
                "uv run ideafindr doctor"
            )
        else:
            log.info("embeddings: configured api endpoint did not answer, trying ollama cloud")

    if b in ("ollama-cloud", "auto") and settings.ollama_api_key:
        if _probe(settings.ollama_base_url, settings.embed_model, settings.ollama_api_key):
            log.info("embeddings: ollama cloud (%s)", settings.embed_model)
            return OpenAICompatEmbedder(
                settings.ollama_base_url, settings.embed_model,
                settings.ollama_api_key, "ollama-cloud",
            )
        if b == "ollama-cloud":
            raise RuntimeError(
                "EMBED_BACKEND=ollama-cloud but /v1/embeddings rejected the key. "
                "Ollama Cloud does not serve embeddings; use EMBED_BACKEND=api with "
                "a provider that does. Run: uv run ideafindr doctor"
            )
        log.info("embeddings: ollama cloud serves no embeddings, falling back to TF-IDF")

    if b == "ollama-local":
        local = f"{settings.ollama_local_url.rstrip('/')}/v1"
        if _probe(local, settings.embed_model):
            log.info("embeddings: local ollama (%s)", settings.embed_model)
            return OpenAICompatEmbedder(local, settings.embed_model, "", "ollama-local")
        raise RuntimeError(f"No local ollama at {settings.ollama_local_url}")

    log.info("embeddings: TF-IDF + SVD (in-process, offline)")
    return TfidfEmbedder()
