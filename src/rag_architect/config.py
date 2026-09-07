from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ChunkingSettings(BaseModel):
    """Document splitting, measured in tokens rather than characters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=512)
    encoding_name: str = Field(default="cl100k_base")

    @model_validator(mode="after")
    def check_overlap_smaller_than_size(self) -> Self:
        """
        Cross-field invariant: two individually legal values can still form an
        absurd combination (every chunk fully overlapping the previous one, so
        the cursor never advances). Field constraints validate each field in
        isolation and cannot catch this.
        """
        if self.chunk_overlap >= self.chunk_size:
            msg = (
                f"chunk_overlap ({self.chunk_overlap}) must be strictly smaller "
                f"than chunk_size ({self.chunk_size})"
            )
            raise ValueError(msg)
        return self


class EmbeddingSettings(BaseModel):
    """Embedding model, served by fastembed (ONNX, no torch)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    batch_size: int = Field(default=32, ge=1, le=256)
    # Cache for ONNX models downloaded on first use. Must be baked into a Docker
    # layer, otherwise every Cloud Run cold start re-downloads the model.
    cache_dir: Path | None = Field(default=None)


class VectorStoreSettings(BaseModel):
    """Qdrant connection and search parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(default="http://localhost:6333")
    api_key: SecretStr | None = Field(default=None)
    collection_name: str = Field(default="pl_reports", min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    timeout_s: float = Field(default=10.0, gt=0)


class LLMSettings(BaseModel):
    """
    LLM client behind an OpenAI-compatible API.

    Defaults point at a local Ollama instance. Switching to production (Groq)
    only requires two environment variables, RAG_LLM__BASE_URL and
    RAG_LLM__API_KEY: no code changes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = Field(default="http://localhost:11434/v1")
    model: str = Field(default="qwen2.5:7b")
    api_key: SecretStr | None = Field(default=None)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    timeout_s: float = Field(default=60.0, gt=0)


class APISettings(BaseModel):
    """HTTP exposure of the application."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: SecretStr | None = Field(default=None)


class Settings(BaseSettings):
    """
    Application-wide configuration, validated once at startup (fail-fast).

    Composed of narrow sub-models so that each component receives only what it
    needs: the chunker is handed a ChunkingSettings and can never reach the LLM
    API key. Assembly happens in one place (the CLI, later the FastAPI
    lifespan), which is dependency injection without a framework.

    Environment variables are prefixed and nested:

        RAG_CHUNKING__CHUNK_SIZE=1024
        RAG_LLM__BASE_URL=https://api.groq.com/openai/v1
        RAG_VECTOR_STORE__API_KEY=...

    `extra="forbid"` is set on this model *and* on every sub-model: for a nested
    field it is the sub-model's own setting that rejects a misspelled variable,
    from both the environment and the .env file.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
    )

    app_name: str = Field(default="LLM-RAG-Architect")
    environment: Environment = Field(default=Environment.DEVELOPMENT)

    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)


@lru_cache
def get_settings() -> Settings:
    """
    Build the Settings instance once and cache it.

    The cache makes this a shared singleton; `frozen=True` on Settings and on
    every sub-model is what keeps that singleton a constant rather than a
    mutable global. Tests must call `get_settings.cache_clear()` between cases.
    """
    return Settings()
