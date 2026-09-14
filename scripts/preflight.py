"""Step 0: probe every external dependency before building on it.

Run:  uv run python scripts/preflight.py
"""

from __future__ import annotations

import asyncio
import sys

import httpx

from ideafindr.config import settings

OK, BAD, WARN = "\033[32m  OK  \033[0m", "\033[31m FAIL \033[0m", "\033[33m WARN \033[0m"
results: dict[str, bool] = {}


def report(name: str, ok: bool, detail: str = "", warn: bool = False) -> None:
    tag = WARN if warn else (OK if ok else BAD)
    print(f"[{tag}] {name:<34} {detail}")
    results[name] = ok


def probe_arctic() -> None:
    b = settings.arctic_base

    async def _probe() -> None:
        # Use the real client, not bare httpx: it retries the data:null throttle
        # response, which a plain request hits often enough to fail a probe that
        # is actually fine.
        from ideafindr.collectors.reddit_arctic import ArcticClient

        async with ArcticClient() as cl:
            try:
                rows = await cl.get("/api/subreddits/search",
                                    {"subreddit_prefix": "coldplunge", "limit": 1})
                report("arctic: subreddits/search", bool(rows), f"{len(rows)} result(s)")
            except Exception as e:  # noqa: BLE001
                report("arctic: subreddits/search", False, repr(e)[:70])

            try:
                rows = await cl.get("/api/posts/search",
                                    {"subreddit": "coldplunge", "query": "chiller", "limit": 2})
                report("arctic: posts/search", bool(rows), f"{len(rows)} post(s)")
            except Exception as e:  # noqa: BLE001
                report("arctic: posts/search", False, repr(e)[:70])

    asyncio.run(_probe())

    # Known constraint: keyword search REQUIRES author or subreddit scope.
    try:
        r = httpx.get(f"{b}/api/posts/search", params={"query": "cold plunge"}, timeout=30)
        expected = r.status_code == 400 and "requires one of" in r.text
        report(
            "arctic: unscoped query rejected",
            expected,
            "constraint confirmed" if expected else f"UNEXPECTED {r.status_code}",
        )
    except Exception as e:
        report("arctic: unscoped query rejected", False, repr(e)[:70])


def probe_ollama() -> None:
    key = settings.ollama_api_key
    if not key:
        report("ollama: API key present", False, "OLLAMA_API_KEY not set in .env", warn=True)
        return
    base = settings.ollama_base_url.rstrip("/")
    h = {"Authorization": f"Bearer {key}"}

    try:
        r = httpx.get(f"{base}/models", headers=h, timeout=30)
        ids = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
        report(
            "ollama: GET /v1/models",
            r.status_code == 200,
            f"{len(ids)} models" if ids else f"HTTP {r.status_code}",
        )
        if ids:
            print("           available:", ", ".join(sorted(ids)[:18]))
    except Exception as e:
        report("ollama: GET /v1/models", False, repr(e)[:70])

    try:
        r = httpx.post(
            f"{base}/chat/completions",
            headers=h,
            json={
                "model": settings.fast_model,
                "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
                # Reasoning models (gpt-oss, glm, deepseek) spend tokens on
                # internal reasoning first; too small a budget returns an empty
                # message and looks like a failure.
                "max_tokens": 512,
            },
            timeout=120,
        )
        ok = r.status_code == 200
        msg = (
            (r.json()["choices"][0]["message"]["content"] or "")[:40].strip() or "(empty reply)"
            if ok
            else f"HTTP {r.status_code}: {r.text[:90]}"
        )
        report(f"ollama: chat ({settings.fast_model})", ok and msg != "(empty reply)", msg)
    except Exception as e:
        report("ollama: chat/completions", False, repr(e)[:70])

    # THE decisive probe: if this fails we cluster with TF-IDF instead.
    try:
        r = httpx.post(
            f"{base}/embeddings",
            headers=h,
            json={"model": settings.embed_model, "input": "hello world"},
            timeout=60,
        )
        ok = r.status_code == 200
        dim = len(r.json()["data"][0]["embedding"]) if ok else 0
        report(
            "ollama: /v1/embeddings (CLOUD)",
            ok,
            f"dim={dim}" if ok else f"HTTP {r.status_code} -- not served by Ollama Cloud",
            warn=not ok,
        )
        if not ok:
            print("           -> expected: this stack is cloud-only and clusters "
                  "with TF-IDF.")
            print("           -> `report --engine gpt-researcher` and `ask` are "
                  "unavailable; all")
            print("              other commands are unaffected. See the README.")
    except Exception as e:
        report("ollama: /v1/embeddings (CLOUD)", False, repr(e)[:70], warn=True)


def probe_embedder() -> None:
    """Report which embedding backend the pipeline will actually use."""
    from ideafindr.embed import get_embedder

    try:
        name = get_embedder().name
        report(f"embeddings backend: {name}", True,
               "in-process, no API" if name == "tfidf" else "API-backed")
    except Exception as e:  # noqa: BLE001
        report("embeddings backend", False, repr(e)[:70])


if __name__ == "__main__":
    print("\n=== Arctic Shift (Reddit) ===")
    probe_arctic()
    print("\n=== Ollama Cloud ===")
    probe_ollama()
    print("\n=== Embeddings ===")
    probe_embedder()

    hard = ["arctic: subreddits/search", "arctic: posts/search"]
    failed = [k for k in hard if not results.get(k)]
    print()
    if failed:
        print(f"BLOCKING failures: {failed}")
        sys.exit(1)
    print("Core data source reachable.")
