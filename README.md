# Ideafindr

Find out what people are **actually saying** about any topic, and turn it into a
report a marketing team can act on.

Combines a social-data collection layer (Reddit via Arctic Shift, open web,
optionally TikTok and Instagram) with a quantitative analysis layer (theme
clustering, momentum, share of voice, language bank) and
[gpt-researcher](https://github.com/assafelovic/gpt-researcher) /
[deep-searcher](https://github.com/zilliztech/deep-searcher) for written
synthesis. All LLM calls go to **Ollama Cloud**.

```
ideafindr research "cold plunge tubs"
```

---

## How the pieces fit

The upstream tools don't compose directly: collectors emit platform-specific
JSON, while research agents expect *a search engine*. The keystone is a small
**bridge** — a local HTTP endpoint that serves our scraped corpus in the exact
shape gpt-researcher's `RETRIEVER=custom` expects:

```json
[{"url": "https://reddit.com/...", "raw_content": "post + top comments"}]
```

That tiny contract means gpt-researcher's whole agent loop — query planning,
source curation, citation, report writing — runs over scraped social data with
**no fork of the upstream repo**.

```
topic → plan (LLM) → collect → SQLite+FTS → cluster → label (LLM) → signals → report
                                     │
                                     └→ bridge (/retrieve) → gpt-researcher → summary
                                     └→ export → deep-searcher → follow-up Q&A
```

| Upstream tool | Where it's used |
|---|---|
| Reddit API | `collectors/reddit_api.py` — 100 queries/min, used when credentials exist |
| `arctic_shift` | `collectors/reddit_arctic.py` — free Reddit archive, the fallback |
| DDG dorking | `collectors/dork.py` — `site:reddit.com` snippets, never touches Reddit |
| HN (Algolia) | `collectors/hackernews.py` — free, no key, technical skew |
| Lemmy | `collectors/lemmy.py` — federated, free; tech topics only in practice |
| Stack Exchange | `collectors/stackexchange.py` — free, 300/day, planner picks the sites |
| `gpt-researcher` | `engines/researcher.py` — wired and tested, but needs an embeddings API (see Setup) |
| `deep-searcher` | `engines/searcher.py` — same; `ideafindr ask` |
| `TikTok-Api` | `collectors/tiktok.py` — optional, off by default |
| `instaloader` | `collectors/instagram.py` — optional, off by default |

**Not included, deliberately.** `stanford-oval/storm` and `dzhng/deep-research`
run the same plan→search→synthesize loop as gpt-researcher; running three of
them is duplicated work for one report section. `xdevplatform/xurl` is deferred
because meaningful X search needs a paid API tier — the `Collector` protocol in
`collectors/base.py` is where it drops in when that budget exists.

---

## Setup

```bash
uv sync --extra dev          # everything you need, plus the test suite
uv sync --extra engines      # + gpt-researcher / deep-searcher (needs an embeddings provider)
uv sync --extra scrapers     # + TikTok and Instagram (read the warning below)

cp .env.example .env         # then add your Ollama Cloud key
uv run python scripts/preflight.py
```

`preflight.py` probes every external dependency and prints a pass/fail line per
endpoint. **Run it first** — it answers questions that change how the pipeline
behaves, in particular whether your Ollama Cloud key authorises embeddings.

### Embeddings

Nothing runs a model on this machine. Chat goes to Ollama Cloud; embeddings come
from a remote provider or not at all.

**Ollama Cloud serves no embeddings.** `/v1/embeddings` returns
`404 path not found` and `/api/embed` returns `401 unauthorized` for cloud keys
(verified 2026-09-12). Chat works fine.

So by default clustering uses **TF-IDF + SVD** — pure scikit-learn, in-process,
instant, no API key. It is a real implementation, not a stub: it clusters short
social posts well enough to produce usable themes. Its weakness is paraphrase —
it cannot tell that "my chiller is deafening" and "the compressor keeps the
neighbours up" are the same complaint, because they share no words.

**What that costs you:** `report --engine gpt-researcher` and `ask` do not work.
Both build their own vector stores and need a reachable embeddings API; neither
can use an in-process embedder. They fail with an explicit message saying so.
Everything else — `plan`, `collect`, `analyze`, `report`, `runs` — is unaffected,
and the report keeps every section except the LLM-written executive summary.

**To turn on a real embedder**, point `EMBED_BASE_URL` / `EMBED_API_KEY` at any
OpenAI-compatible `/v1/embeddings` endpoint. Jina and Voyage have free tiers;
OpenAI, Cohere and Together also work. No code change — one class
(`OpenAICompatEmbedder`) covers every provider:

```bash
EMBED_BACKEND=auto
EMBED_BASE_URL=https://api.jina.ai/v1
EMBED_API_KEY=jina_...
EMBED_MODEL=jina-embeddings-v3
```

`EMBED_BACKEND=auto` probes your provider, then Ollama Cloud (in case it ever
ships embeddings), then falls back to TF-IDF. `ollama-local` exists but is
deliberately **not** in the `auto` chain: a daemon appearing on this box must not
silently capture embedding work on hardware that cannot afford it. Name it
explicitly if you are on a bigger machine.

### Models

Ollama Cloud lists model IDs **without** a `-cloud` suffix on the `/v1`
endpoint. See the live list with `ideafindr doctor`.

Both `FAST_MODEL` and `SMART_MODEL` default to **`gpt-oss:120b`**, and the second
one is deliberate. Several models on this endpoint are reasoning models that emit
a `reasoning` field before any `content` and can burn the entire `max_tokens`
budget doing it — returning HTTP 200 with an **empty message**. Downstream that
surfaces as `NoneType has no len()` from inside gpt-researcher, which tells you
nothing about the cause.

Measured on a 600-word request at `max_tokens=2000`:

| model | time | content | finish | reasoning |
|---|---|---|---|---|
| `gpt-oss:120b` | 6.2s | 7020 chars | stop | 443 chars |
| `deepseek-v4-pro:0813` | 14.3s | 5546 chars | stop | 1433 chars |
| `minimax-m3` | 6.5s | 4271 chars | stop | 1054 chars |
| `qwen3.5:397b` | 25.2s | 4476 chars | stop | 4332 chars |
| `mistral-large-3:675b` | 29.3s | 8427 chars | **length** (truncated) | 0 |
| `glm-5.3` | 32.5s | **0 chars** | length | 9541 chars |
| `kimi-k3` | — | **0 chars** | — | — |

`ideafindr.llm.chat()` raises `EmptyCompletion` naming the model, the
`finish_reason` and the reasoning length rather than passing an empty string
along, so if you switch models you find out immediately.

> **Why a fallback chain at all?** It was written on Chimera Linux (musl),
> where `fastembed` and `sentence-transformers` are not installable —
> `onnxruntime` and PyTorch ship glibc-only wheels. The project now runs on
> glibc, so that particular constraint is gone, but the chain stays: the
> deployment target is low-end hardware where running a local model is the wrong
> trade regardless of what pip will install. TF-IDF remains a real
> implementation rather than a stub.

---

## Using it

`ideafindr` with no arguments opens the TUI, which is the primary interface.

```bash
ideafindr
```

Three screens, each doing one thing.

**Home** is a prompt and your past projects, and nothing else:

```
                          IDEAFINDR
             what people say · what people search

   ┌────────────────────────────────────────────────────┐
   │ Research a topic…                                  │
   └────────────────────────────────────────────────────┘
    enter to research · ↑↓ then enter to open · x to delete

   Projects
    cold plunge tubs    716 docs · 19 themes · 12 intents · Sep 13
    cold plunge tubs    673 docs · 18 themes · Sep 12
```

`↓` from the prompt moves into the project list, `esc` goes back. `x` (or
`delete`) removes the highlighted project after confirming — its documents,
themes, search demand and rendered reports. There is a `ideafindr delete <run>`
subcommand for the same thing.

Typing a topic runs the whole pipeline — plan, collect, cluster, harvest search
demand, write the report — on a **Research** screen that shows each stage as it
happens, with the pipeline's own log underneath:

```
   Researching “cold plunge tubs”

     ✓  Plan the sweep          12 subreddits, 8 keywords
     ✓  Collect the corpus      716 documents
     ⠿  Cluster into themes
     ·  Harvest search demand
     ·  Write the report

   ┌──────────────────────────────────────────────────────────┐
   │ 20:45:36  discovered r/coldplunge(13,000), r/becoming…    │
   │ 20:45:36  collector reddit returned 681 documents         │
   └──────────────────────────────────────────────────────────┘
```

Each past project opens as its **own page** — themes, search demand and full-text
search over that corpus, with `esc` to go back:

| key | on a project page |
|---|---|
| `s` | search this project's corpus |
| `a` | re-cluster into themes |
| `d` | harvest search demand |
| `r` | write the report |
| `esc` | back to home |
| `^t` | toggle light/dark |

The app uses your terminal's own 16-colour palette rather than hardcoded colours,
so it matches whatever scheme you already run.

### Commands

Every stage is also a subcommand, for scripting and for anywhere a full-screen
app is the wrong shape:

```bash
ideafindr plan    "topic"                 # show the collection plan, collect nothing
ideafindr collect "topic" --days 180      # build a corpus
ideafindr analyze [run]  [-k 15]          # cluster → label → quotes → signals
ideafindr research "topic"                # all of the above, end to end
ideafindr demand  [--reuse]               # harvest search demand, find gaps
ideafindr runs                            # list past runs
ideafindr tui                             # same as bare `ideafindr`
ideafindr bridge                          # run the retriever endpoint standalone
ideafindr doctor                          # re-run the preflight probes

# needs an embeddings provider (see Setup) -- unavailable with the TF-IDF default:
#   ideafindr report [run] --engine gpt-researcher
#   ideafindr ask "question" --run <run>
```

Reports land in `data/reports/<run>/` as `report.md` and a self-contained
`report.html` (inline CSS and SVG, no CDN — it survives being emailed and opened
from disk).

---

## What's in a report

1. **Theme map** — themes ranked by volume × momentum, with trend sparklines
3. **Voice of the customer** — verbatim quotes, never paraphrased, each with a
   live permalink
4. **Language bank** — the words and phrases people actually use, for ad copy
5. **Share of voice** — brand mentions and their trend
6. **Methodology** — sources, counts, date range, and the caveats below

**Momentum** is a theme's share of the last 30 days divided by its share of the
whole window. `>1` means rising faster than the topic overall. It's deliberately
a *share* rather than raw recent volume, which would just track how much got
collected.

---

## Things worth knowing before you sell this to a client

- **Reddit skews the picture.** Younger, more male, more English-speaking, more
  technical than the general market. The report says so in its methodology
  section; leave that in.
- **Arctic Shift is one person's free service** with no uptime guarantee. The
  client is polite by default (~2 req/s, backoff on 429). Don't raise
  `ARCTIC_RPS`. If it becomes load-bearing, move to the bulk `.zst` dumps.
- **Web articles carry their collection date, not publication date** —
  trafilatura's date extraction is unreliable often enough that trusting it
  would corrupt the trend charts. Web docs feed themes and quotes but not
  time-series signals.
- **TikTok and Instagram scraping violates those platforms' Terms of Service.**
  For a company selling research to clients that's commercial and legal
  exposure, not just a technical risk. Both are off by default and every failure
  is swallowed so a dead scraper degrades the corpus instead of killing the run.
  Worth a legal read before anything ships to a client.
  - Instagram specifically is close to non-viable at volume: hashtag browsing is
    login-gated and anonymous requests hit HTTP 429 within a handful of calls.
    The default `websearch` mode finds public post URLs via search instead, which
    yields less but doesn't get accounts banned.
- **gpt-researcher is a fast-moving dependency.** It's pinned. If a report comes
  back citing only web sources, check that `RETRIEVER_ENDPOINT` is still the env
  var name upstream reads — that's the contract that can break silently.

---

## Development

```bash
uv run pytest                # offline: collector tests use recorded HTTP
uv run ideafindr doctor      # live: probes Arctic Shift and Ollama Cloud
```

Tests never hit Arctic Shift. They pin the live-API behaviours that actually
broke things during development:

- a `200` response carrying `"data": null` — throttling, must be retried, not
  read as empty
- `"Timeout. Maybe slow down a bit"` — the query was too expensive, so the fix
  is a smaller page, not a longer wait
- server-side full-text search failing while plain listing succeeds — the slice
  is recovered by fetching unfiltered and matching the keyword locally, because
  shrinking the page can't help when the cost is the text search itself
