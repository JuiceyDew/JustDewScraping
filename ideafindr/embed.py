"""Embedding backends, chosen at runtime.

This machine is Chimera Linux (musl libc). `onnxruntime` and PyTorch publish
glibc-only wheels, so fastembed / sentence-transformers are NOT installable --
hence a backend chain rather than one hard dependency:

    ollama-cloud  -> if the cloud key authorizes /v1/embeddings
    ollama-local  -> if a local ollama daemon is running
    tfidf         -> pure scikit-learn, always works, no API, no network

TF-IDF+SVD is a real fallback, not a stub: for short social posts that share
vocabulary it clusters decently. It is weaker at grouping posts that mean the
same thing in different words, which is exactly what the LLM labelling step
partially compensates for.
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


class OllamaEmbedder(Embedder):
    """OpenAI-compatible /v1/embeddings, cloud or local."""

    def __init__(self, base_url: str, model: str, api_key: str = "", name: str = "ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name
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
    """TF-IDF -> TruncatedSVD (LSA). Pure sklearn: works on musl, offline, free."""

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
    """Resolve the backend. `auto` probes cloud, then local, then falls back."""
    b = (backend or settings.embed_backend).lower()

    if b in ("ollama-cloud", "auto") and settings.ollama_api_key:
        if _probe(settings.ollama_base_url, settings.embed_model, settings.ollama_api_key):
            log.info("embeddings: ollama cloud (%s)", settings.embed_model)
            return OllamaEmbedder(
                settings.ollama_base_url, settings.embed_model,
                settings.ollama_api_key, "ollama-cloud",
            )
        if b == "ollama-cloud":
            raise RuntimeError(
                "EMBED_BACKEND=ollama-cloud but /v1/embeddings rejected the key. "
                "Run: uv run python scripts/preflight.py"
            )
        log.info("embeddings: ollama cloud rejected /v1/embeddings, trying local")

    if b in ("ollama-local", "auto"):
        local = f"{settings.ollama_local_url.rstrip('/')}/v1"
        if _probe(local, settings.embed_model):
            log.info("embeddings: local ollama (%s)", settings.embed_model)
            return OllamaEmbedder(local, settings.embed_model, "", "ollama-local")
        if b == "ollama-local":
            raise RuntimeError(f"No local ollama at {settings.ollama_local_url}")
        log.info("embeddings: no local ollama, falling back to TF-IDF")

    log.info("embeddings: TF-IDF + SVD (musl-safe, offline)")
    return TfidfEmbedder()
