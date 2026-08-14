import pytest
from pydantic import ValidationError
from src.config import Environment, Settings


def test_settings_load_from_env(monkeypatch):
    """Vérifie le chargement des variables d'environnement."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123456")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("QDRANT_PORT", "6333")

    test_settings = Settings()

    assert test_settings.openai_api_key == "sk-test-123456"
    assert test_settings.environment == Environment.PRODUCTION
    assert test_settings.qdrant_port == 6333


def test_settings_missing_api_key(monkeypatch):
    """Vérifie la levée d'erreur si la clé API est manquante."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings()