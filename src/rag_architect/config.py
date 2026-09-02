from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Environnement de déploiement."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Configuration globale de l'application.
    Validation stricte au démarrage (Fail-Fast).
    """

    app_name: str = Field(default="LLM-RAG-Architect")
    environment: Environment = Field(default=Environment.DEVELOPMENT)

    # Base de données vectorielle
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)

    # Clé API OpenAI obligatoire (plante si absente de l'environnement)
    openai_api_key: str = Field(..., description="Clé d'API OpenAI pour les embeddings et LLM")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Crée l'instance Settings une seule fois et la met en cache.
    C'est le pattern standard de FastAPI (Dependency Injection).
    """
    return Settings()
