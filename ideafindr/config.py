"""Settings. One place that knows about Ollama Cloud, so every consumer
(our code, gpt-researcher, deep-searcher) gets wired the same way.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ideafindr.state import resolve_state_dir

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Precedence: init > environment > settings.json > .env > secrets.

        `settings.json` is what the web UI writes (see ideafindr/state.py). It
        sits below the real environment on purpose: an operator, or a NixOS
        `EnvironmentFile`, must always be able to override whatever the UI
        stored. Earlier sources in this tuple win.
        """
        from pydantic_settings import JsonConfigSettingsSource

        from ideafindr.state import settings_file

        path = settings_file()
        if path.exists():
            json_source = JsonConfigSettingsSource(settings_cls, json_file=path)
            return (init_settings, env_settings, json_source,
                    dotenv_settings, file_secret_settings)
        return (init_settings, env_settings, dotenv_settings, file_secret_settings)

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

    # X / Twitter. No API key: twscrape authenticates with the browser's session
    # cookies. `auth_token` identifies the account; `ct0` is the CSRF token the
    # GraphQL calls require. Use a dedicated burner account -- never a personal
    # one. Entered through the web UI and stored in the state dir, not here.
    x_auth_token: str = ""
    x_ct0: str = ""

    # Instagram. A logged-in sessionid is what lets instaloader reach hashtag
    # feeds; without it the collector falls back to public-post web search.
    instagram_session_user: str = ""
    instagram_sessionid: str = ""
    instagram_csrftoken: str = ""

    enable_tiktok: bool = False
    enable_instagram: bool = False
    tiktok_ms_token: str = ""

    # --- demand ---------------------------------------------------------------
    # Autocomplete endpoints are undocumented and unmetered; 3/s is polite enough
    # to stay welcome and fast enough that a full sweep finishes in a minute or two.
    demand_rps: float = 3.0
    demand_max_queries: int = 1200

    # --- web UI ---------------------------------------------------------------
    # 0.0.0.0 binds every interface so the UI is reachable from the LAN. There is
    # no app auth (see ideafindr/web/app.py), so the UI holds the Ollama key and
    # live social cookies, and anyone who can reach the port can start research
    # or read settings. The NixOS module only opens the firewall when asked; if
    # you expose this, restrict the port to your LAN or put a proxy/VPN in front.
    #
    # 0.0.0.0 rather than the machine's LAN IP on purpose: the address here comes
    # from DHCP and changes, a bind to a stale IP fails silently on reboot.
    web_host: str = "0.0.0.0"
    web_port: int = 8000

    # --- bridge ---------------------------------------------------------------
    # Stays on localhost: it serves the raw scraped corpus to gpt-researcher and
    # has no business being reachable from the network. Do not widen this.
    bridge_host: str = "127.0.0.1"
    bridge_port: int = 8899

    # --- llm budget -----------------------------------------------------------
    # Ceiling on completion tokens for a single chat call. Kept modest because
    # several Ollama Cloud models burn the whole budget on a `reasoning` field
    # before writing any content; see ideafindr/llm.py.
    llm_max_tokens: int = 2048

    # --- paths ----------------------------------------------------------------
    # All state lives under the state dir, which defaults to ../data (gitignored)
    # and is overridden by IDEAFINDR_STATE_DIR (the NixOS module sets this to the
    # systemd StateDirectory). Resolved at instantiation so a relocated state dir
    # moves the database, reports and sessions together.
    db_path: Path = DATA_DIR / "ideafindr.db"
    reports_dir: Path = DATA_DIR / "reports"
    sessions_dir: Path = DATA_DIR / "sessions"

    @model_validator(mode="after")
    def _relocate_paths(self) -> "Settings":
        """Move the default paths when IDEAFINDR_STATE_DIR relocates the state dir.

        The field defaults are anchored at `ROOT/data`. If the operator pointed
        the state dir somewhere else, the database, reports and sessions follow
        it -- but a path set explicitly via env/json/dotenv is left alone.
        """
        state = resolve_state_dir()
        if state == DATA_DIR:
            return self
        for name, leaf in (("db_path", "ideafindr.db"),
                           ("reports_dir", "reports"),
                           ("sessions_dir", "sessions")):
            if getattr(self, name) == DATA_DIR / leaf:
                object.__setattr__(self, name, state / leaf)
        return self

    @property
    def bridge_url(self) -> str:
        return f"http://{self.bridge_host}:{self.bridge_port}"

    @property
    def web_url(self) -> str:
        """A human-facing URL. 0.0.0.0 is a bind address, not a destination, so
        it is shown as localhost -- the operator reaches it via the machine's
        real IP or hostname."""
        shown = "127.0.0.1" if self.web_host in ("0.0.0.0", "::") else self.web_host
        return f"http://{shown}:{self.web_port}"

    def ensure_dirs(self) -> None:
        for p in (self.db_path.parent, self.reports_dir, self.sessions_dir):
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
