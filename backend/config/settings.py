"""
config/settings.py
------------------
Central configuration loaded from environment variables / .env file.
Every module imports `settings` from here — no API keys in source code.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── LLM ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")

    # ── Directories ─────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"
    UPLOAD_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "uploads"
    CHROMA_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "chroma_db"

    # ── Cache ───────────────────────────────────────────────────────────────
    CACHE_MAX_SIZE: int = Field(default=200, env="CACHE_MAX_SIZE")   # max cached entries
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")  # 1 hour

    # ── Memory ──────────────────────────────────────────────────────────────
    MEMORY_MAX_MESSAGES: int = Field(default=10, env="MEMORY_MAX_MESSAGES")

    class Config:
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton — import this everywhere
settings = Settings()

# Ensure required directories exist at startup
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", settings.BASE_DIR)
print("ENV file expected at:", settings.BASE_DIR / ".env")
print("API key loaded:", bool(settings.OPENAI_API_KEY))
print("API key starts with:", settings.OPENAI_API_KEY[:7] if settings.OPENAI_API_KEY else "NOT FOUND")
