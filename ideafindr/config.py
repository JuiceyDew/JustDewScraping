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
    # Cloud-only deployment: chat goes to Ollama Cloud, and clustering uses the
    # in-process TF-IDF embedder. There is no embeddings API in the stack because
    # Ollama Cloud does not serve one -- /v1/embeddings returns 404 and /api/embed
    # returns 401 for cloud keys. "auto" still probes cloud -> local -> tfidf and
    # would pick up a cloud embeddings endpoint automatically if one appeared.
    embed_backend: str = "tfidf"
    # Only consulted when embed_backend points at an actual embeddings API.
    embed_model: str = "all-minilm"
    ollama_local_url: str = "http://localhost:11434"

    # --- sources --------------------------------------------------------------
    arctic_base: str = "https://arctic-shift.photon-reddit.com"
    arctic_rps: float = 2.0  # be a good citizen: it's one person's free service
    enable_tiktok: bool = False
    enable_instagram: bool = False
    tiktok_ms_token: str = ""
    instagram_session_user: str = ""

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
