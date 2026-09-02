import pytest
from pydantic import ValidationError

from rag_architect.config import Environment, Settings

CONFIG_ENV_VARS = ("APP_NAME", "ENVIRONMENT", "QDRANT_HOST", "QDRANT_PORT", "OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch, tmp_path):
    """
    Rend chaque test hermétique vis-à-vis de la machine hôte.

    - `chdir` vers un dossier vide : `env_file=".env"` est un chemin relatif,
      il ne résout donc plus sur le .env du dépôt.
    - suppression des variables lues par Settings : on part d'un état connu,
      qu'elles soient définies dans le shell du dev ou dans le runner CI.
    """
    monkeypatch.chdir(tmp_path)
    for var in CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_settings_load_from_env(monkeypatch):
    """Les variables d'environnement alimentent bien les champs."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123456")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("QDRANT_PORT", "7777")

    settings = Settings()

    assert settings.openai_api_key == "sk-test-123456"
    assert settings.environment is Environment.PRODUCTION
    assert settings.qdrant_port == 7777


def test_settings_defaults_apply_when_env_is_empty():
    """Sans variable définie, les valeurs par défaut s'appliquent."""
    settings = Settings(openai_api_key="sk-test-123456")

    assert settings.environment is Environment.DEVELOPMENT
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333


def test_settings_missing_api_key():
    """L'absence de clé API fait échouer le démarrage (fail-fast)."""
    with pytest.raises(ValidationError):
        Settings()
