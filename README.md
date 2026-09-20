# Ideafindr

Find out what people are **actually saying** about any topic, and turn it into a
report a marketing team can act on.

Combines a social-data collection layer (Reddit via Arctic Shift or the official
API, X/Twitter, Bluesky, Instagram, the open web, more) with a quantitative
analysis layer (theme clustering, momentum, share of voice, language bank) and
[gpt-researcher](https://github.com/assafelovic/gpt-researcher) /
[deep-searcher](https://github.com/zilliztech/deep-searcher) for written
synthesis. Chat goes to **Ollama Cloud**; embeddings are a hosted provider or
in-process TF-IDF.

The interface is a small **local web UI** (no build step, no CDN, no app auth),
served from a Nix flake for a homelab.

```
ideafindr              # serves http://0.0.0.0:8000 (reachable on the LAN)
ideafindr research "cold plunge tubs"   # or drive it from the CLI
```

---

## Design

```
topic → plan (LLM) → collect → SQLite+FTS → cluster → label (LLM) → signals → report
                                     │
                                     ├→ bridge (/retrieve) → gpt-researcher → summary
                                     └→ export → deep-searcher → follow-up Q&A
```

The **bridge** is the keystone: a local HTTP endpoint that serves the scraped
corpus in the exact shape gpt-researcher's `RETRIEVER=custom` expects —

```json
[{"url": "https://reddit.com/...", "raw_content": "post + top comments"}]
```

— so the whole upstream agent loop runs over scraped social data with **no fork**.

| Component | Where |
|---|---|
| Web UI | `ideafindr/web/` — FastAPI + Jinja, server-rendered |
| Runtime state / secrets | `ideafindr/state.py` — `settings.json`, mode 0600, outside the repo |
| Reddit (official API) | `collectors/reddit_api.py` — 100 queries/min; needs Reddit approval |
| Reddit (Arctic Shift) | `collectors/reddit_arctic.py` — free archive, the fallback |
| X / Twitter | `collectors/x_twitter.py` — twscrape, cookie-based, no API key |
| Bluesky | `collectors/bluesky.py` — free, no key |
| Instagram | `collectors/instagram.py` — instaloader; sessionid enables hashtags |
| Web / dork / HN / Lemmy / Stack Exchange | `collectors/` — free, no key |
| `gpt-researcher` | `engines/researcher.py` — needs an embeddings API |
| `deep-searcher` | `engines/searcher.py` — same; `ideafindr ask` |

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone git@github.com:JuiceyDew/JustDewScraping.git
cd JustDewScraping

uv sync --extra dev
cp .env.example .env
uv run ideafindr web        # serves http://0.0.0.0:8000
```

HTTPS works too if you have not set up an SSH key:

```bash
git clone https://github.com/JuiceyDew/JustDewScraping.git
```

Then, in the UI:

1. **Settings** — paste your Ollama Cloud key. It is written to
   `data/settings.json` (mode 0600) and never to the repo.
2. **Credentials** — add X and/or Instagram cookies (below).
3. Type a topic on the home page and hit Research.

Run `uv run ideafindr doctor` to probe every external dependency.

---

## Credentials

The UI has no authentication and holds API keys and live session cookies. It
binds `0.0.0.0:8000` (`WEB_HOST`/`WEB_PORT`, or `ideafindr web --host`) so it is
reachable on the LAN — but anyone on that network can start a run, read the
Settings/Credentials pages, or delete projects. **Restrict the firewall to your
subnet, or put a VPN / authenticating reverse proxy in front of it.** The
retriever bridge stays on `127.0.0.1`. It stores nothing in the repository.

### The safe way: import from your browser

The UI runs on the same machine you log into, so the Credentials page can read
that browser's cookie store directly. You log in normally; the platform never
sees automation.

On Linux, **Firefox** cookies decrypt without an external keyring — it is the
happy path. Chromium-family reads need the OS keyring, which a headless systemd
service may not have unlocked, in which case paste manually.

### Manual paste

Every platform has exact steps on the Credentials page. In short: open DevTools
(F12) → Application → Cookies → copy the values.

| Platform | Cookies | Notes |
|---|---|---|
| X / Twitter | `auth_token`, `ct0` | **Dedicated burner account.** No proxy: single account, low volume. |
| Instagram | `sessionid`, `csrftoken` | Login-gated and rate-limits hard; expect low yield. |
| Reddit | client id + secret | Optional; see the licensing note below. |

> **Use burner accounts for X and Instagram — never a personal or client
> account.** Scraping both is against their Terms of Service, which is a
> commercial and legal exposure when you sell the report, not just a technical
> risk.

### Reddit access

Reddit's own API is ~50× faster than Arctic Shift and does not time out, and the
collector is written and tested. But credentials are no longer self-service:
Reddit's [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy)
(November 2025) closed instant signup, and the policy prohibits commercial
resale of Reddit data. Without credentials, `--sources reddit` uses Arctic Shift.
Three sources need no permission:

| source | what it gets |
|---|---|
| `dork` | Reddit threads via `site:reddit.com` search snippets — never contacts Reddit |
| `hackernews` | free, no key, officially provided via Algolia; technical skew |
| `lemmy` | federated and free; in practice only worth it for tech topics |
| `stackexchange` | free, 300 requests/day; the planner picks the sites |

---

## Embeddings

Nothing runs a model on this machine. Chat goes to Ollama Cloud; embeddings come
from a remote provider or not at all.

**Ollama Cloud serves no embeddings.** `/v1/embeddings` returns `404` and
`/api/embed` returns `401` for cloud keys. Chat works fine.

By default clustering uses **TF-IDF + SVD** — pure scikit-learn, in-process,
instant, no API key. It is a real implementation, not a stub, but it cannot tell
that "my chiller is deafening" and "the compressor keeps the neighbours up" are
the same complaint.

**What that costs you:** `report --engine gpt-researcher` and `ask` do not work.
Both build their own vector stores and need a reachable embeddings API.
Everything else is unaffected.

To enable a real embedder, set `EMBED_BASE_URL` / `EMBED_API_KEY` in Settings to
any OpenAI-compatible `/v1/embeddings` endpoint (Jina, Voyage, OpenAI, Cohere,
Together all work). One class covers every provider — no code change.

---

## The LLM is used deliberately sparingly

Ollama Cloud calls are batched and cached so a run is cheap and fast:

- **One batched call labels every theme**, not one call per theme. A 19-theme
  run used to make 19 sequential requests; it now makes two, and the batch is
  chunked so the prompt stays bounded.
- **Responses are cached** in-process, keyed on the exact request, so
  re-analysing an unchanged corpus costs nothing. Disable with
  `IDEAFINDR_LLM_CACHE=0`.
- **`LLM_MAX_TOKENS` caps the completion budget** (`llm_max_tokens`, default
  2048). Several models on this endpoint spend the whole budget on a `reasoning`
  field and return empty content; `ideafindr.llm.chat()` raises
  `EmptyCompletion` naming the model and `finish_reason` when that happens.
- Only **planning**, **theme labelling** and **demand labelling** call the model.
  Clustering, momentum, share of voice and the report body are all local.

Both `FAST_MODEL` and `SMART_MODEL` default to **`gpt-oss:120b`** — the tested
model. Several alternatives (glm-5.3, kimi-k3) return empty content on long
generations.

---

## Using the CLI

Every stage is also a subcommand, for scripting and CI:

```bash
ideafindr web                             # the web UI (same as bare `ideafindr`)
ideafindr plan    "topic"                 # show the collection plan, collect nothing
ideafindr collect "topic" --days 180 --sources reddit,web,bluesky,x
ideafindr analyze [run]  [-k 15]          # cluster → label → quotes → signals
ideafindr research "topic"                # all of the above, end to end
ideafindr demand  [--reuse]               # harvest search demand, find gaps
ideafindr runs                            # list past runs
ideafindr bridge                          # run the retriever endpoint standalone
ideafindr delete  <run>
ideafindr doctor                          # re-run the preflight probes

python simple.py "topic"                  # one-shot, no server
```

Reports land in `<state>/reports/<run>/` as `report.md` and a self-contained
`report.html` (inline CSS and SVG, no CDN — it survives being emailed and opened
from disk).

### What's in a report

Theme map (ranked by volume × momentum, with sparklines) · voice of the customer
(verbatim quotes with permalinks) · search demand and content gaps · language
bank · share of voice · methodology with caveats.

**Momentum** is a theme's share of the last 30 days divided by its share of the
whole window. `>1` means rising faster than the topic overall — a share, not raw
recent volume, which would just track how much was collected.

---

## Nix

The repo is a flake: a package, a dev shell, and a NixOS module.

```nix
{
  inputs.ideafindr.url = "github:JuiceyDew/JustDewScraping";

  services.ideafindr = {
    enable = true;
    host = "0.0.0.0";     # bind all interfaces to reach it from the LAN
    port = 8000;
    openFirewall = true;  # opens only `port`; restrict to your subnet if needed
    # stateDir defaults to /var/lib/ideafindr (systemd StateDirectory, mode 0700)
  };
}
```

The module runs a hardened systemd service with `IDEAFINDR_STATE_DIR` pointed at
its `StateDirectory`. **The Ollama key is not configured in Nix** — enter it in
the web UI and it is stored under the state directory. An `environmentFile` is
supported for anything you would rather set declaratively; the real environment
overrides the UI.

To build from a checkout:

```bash
git clone git@github.com:JuiceyDew/JustDewScraping.git
cd JustDewScraping

nix develop        # dev shell with the test and optional-collector extras
nix build          # the package
nix run . -- --help
```

Or run the flake directly without cloning:

```bash
nix run github:JuiceyDew/JustDewScraping -- --help
```

Verified on x86_64-linux: `nix flake check`, `nix build`, and the packaged web
UI all work, and the test suite passes inside `nix develop`. When installed as a
package, the default state directory is `$XDG_STATE_HOME/ideafindr` (or
`~/.local/state/ideafindr`) because the Nix store is read-only; the NixOS module
overrides this with the systemd `StateDirectory`.

---

## Development

```bash
uv run pytest                # offline: collectors use recorded/mocked HTTP
uv run ideafindr doctor      # live: probes Arctic Shift and Ollama Cloud
```

Tests never hit the network. They pin the behaviours that actually broke during
development:

- a `200` response carrying `"data": null` — throttling, must be retried
- `"Timeout. Maybe slow down a bit"` — smaller page, not a longer wait
- server-side full-text search failing while plain listing succeeds
- PEP 695 `type` aliases are not callable — collectors must pass string literals
  to `Document`, never `Platform("bluesky")`
- batched labelling must tolerate a partial answer without shifting labels

---

## Things worth knowing before you sell this to a client

- **Reddit skews the picture.** Younger, more male, more English-speaking, more
  technical than the general market. The report says so in its methodology.
- **Arctic Shift is one person's free service** with no uptime guarantee. Don't
  raise `ARCTIC_RPS`. If it becomes load-bearing, move to the bulk `.zst` dumps.
- **Web articles carry their collection date, not publication date** —
  trafilatura's date extraction is unreliable, so web docs feed themes and quotes
  but not time-series signals.
- **X, TikTok and Instagram scraping violates those platforms' Terms of
  Service.** Off by default (except X, which needs cookies anyway), and every
  failure is swallowed so a dead scraper degrades the corpus instead of killing
  the run. Worth a legal read before anything ships to a client.
- **gpt-researcher is a fast-moving dependency.** It's pinned. If a report cites
  only web sources, check that `RETRIEVER_ENDPOINT` is still the env var name
  upstream reads.
