"""Settings. One place that knows about Ollama Cloud, so every consumer
(our code, gpt-researcher, deep-searcher) gets wired the same way.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Ollama Cloud (OpenAI-compatible) -------------------------------------
    ollama_api_key: str = ""
    ollama_base_url: str = "https://ollama.com/v1"
    # Model IDs as Ollama Cloud actually lists them (`ideafindr doctor` prints the
    # live list). There is no "-cloud" suffix on the OpenAI-compatible endpoint.
    # cheap model for loop work: planning, labelling, per-doc classification
    fast_model: str = "gpt-oss:120b"
    # Report writing. Also gpt-oss:120b, deliberately: several models on this
    # endpoint (kimi-k3, glm-5.3) are reasoning models that spend the whole
    # max_tokens budget on a `reasoning` field and return EMPTY content on long
    # generations, which surfaces downstream as "NoneType has no len()" inside
    # gpt-researcher. Measured on a 600-word request at max_tokens=2000:
    #   gpt-oss:120b  6.2s  7020 chars  stop
    #   deepseek-v4-pro 14.3s 5546 chars  stop
    #   glm-5.3       32.5s     0 chars  length  (9541 chars of reasoning)
    #   kimi-k3                 0 chars
    smart_model: str = "gpt-oss:120b"

    # --- embeddings -----------------------------------------------------------
    # Embeddings are remote or in-process, never local: this box is low-end, so
    # no model runs here. Ollama Cloud serves chat but no embeddings
    # (/v1/embeddings 404s, /api/embed 401s for cloud keys, verified 2026-09), so
    # a real embedder means a hosted provider -- set embed_base_url and
    # embed_api_key to any OpenAI-compatible endpoint (Jina, Voyage, OpenAI,
    # Cohere, Together). Without one, clustering falls back to in-process TF-IDF,
    # which is fine for social posts and weak for short search queries.
    # "auto" probes api -> ollama-cloud -> tfidf. See ideafindr/embed.py.
    embed_backend: str = "tfidf"
    embed_model: str = "all-minilm"
    # OpenAI-compatible embeddings endpoint, including the /v1 suffix.
    embed_base_url: str = ""
    embed_api_key: str = ""
    # Only used when embed_backend is explicitly "ollama-local"; never by "auto".
    ollama_local_url: str = "http://localhost:11434"

    # --- sources --------------------------------------------------------------
    # Reddit's own API: 100 queries/min with OAuth, versus Arctic Shift's ~2/s
    # that already times out on a fifth of keyword queries. Create a free app at
    # https://reddit.com/prefs/apps (type: script) -- read-only needs only the id
    # and secret, no Reddit password.
    #
    # LICENSING: the free tier is NON-COMMERCIAL (personal, research, bots).
    # Selling reports built on this data needs Reddit's approval and is billed
    # per call. Nothing in the code can detect that line being crossed.
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    # Reddit rejects generic user-agents; it should name the app and its author.
    reddit_user_agent: str = "linux:ideafindr:0.1 (research; by /u/ideafindr)"

    arctic_base: str = "https://arctic-shift.photon-reddit.com"
    arctic_rps: float = 2.0  # be a good citizen: it's one person's free service
    # Stack Exchange allows 300 requests/day/IP unkeyed, 10k with a free key
    # from stackapps.com. Empty is fine; the collector budgets accordingly.
    stackexchange_key: str = ""
    enable_tiktok: bool = False
    enable_instagram: bool = False
    tiktok_ms_token: str = ""
    instagram_session_user: str = ""

    # --- demand ---------------------------------------------------------------
    # Autocomplete endpoints are undocumented and unmetered; 3/s is polite enough
    # to stay welcome and fast enough that a full sweep finishes in a minute or two.
    demand_rps: float = 3.0
    demand_max_queries: int = 1200

    # --- bridge ---------------------------------------------------------------
    bridge_host: str = "127.0.0.1"
    bridge_port: int = 8899

    # --- paths ----------------------------------------------------------------
    db_path: Path = DATA_DIR / "ideafindr.db"
    reports_dir: Path = DATA_DIR / "reports"
    sessions_dir: Path = DATA_DIR / "sessions"

    @property
    def bridge_url(self) -> str:
        return f"http://{self.bridge_host}:{self.bridge_port}"

    def ensure_dirs(self) -> None:
        for p in (DATA_DIR, self.reports_dir, self.sessions_dir):
            p.mkdir(parents=True, exist_ok=True)

    def export_openai_env(self) -> None:
        """Point OpenAI-compatible libraries (gpt-researcher, deep-searcher) at
        Ollama Cloud. They all read these standard vars."""
        os.environ["OPENAI_API_KEY"] = self.ollama_api_key
        os.environ["OPENAI_BASE_URL"] = self.ollama_base_url
        os.environ["FAST_LLM"] = f"openai:{self.fast_model}"
        os.environ["SMART_LLM"] = f"openai:{self.smart_model}"
        os.environ["STRATEGIC_LLM"] = f"openai:{self.smart_model}"


settings = Settings()
