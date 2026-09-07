import os

import pytest
from pydantic import ValidationError

from rag_architect.config import Environment, Settings, get_settings


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch, tmp_path):
    """
    Make every test hermetic with respect to the host machine.

    - `chdir` into an empty directory: `env_file=".env"` is a relative path, so
      it no longer resolves to the repository's .env file.
    - drop every RAG_-prefixed variable, whether it comes from the developer's
      shell or from the CI runner.
    - clear the lru_cache on get_settings, otherwise the first test to call it
      would freeze its own environment for all the following ones.
    """
    monkeypatch.chdir(tmp_path)
    for var in [name for name in os.environ if name.startswith("RAG_")]:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()


def test_defaults_apply_when_env_is_empty():
    """With no variable set, every sub-model falls back to its defaults."""
    settings = Settings()

    assert settings.environment is Environment.DEVELOPMENT
    assert settings.chunking.chunk_size == 512
    assert settings.vector_store.url == "http://localhost:6333"
    assert settings.llm.base_url == "http://localhost:11434/v1"
    assert settings.llm.temperature == 0.0
    assert settings.api.port == 8000


def test_nested_env_vars_populate_submodels(monkeypatch):
    """The RAG_<SECTION>__<FIELD> convention reaches the nested models."""
    monkeypatch.setenv("RAG_CHUNKING__CHUNK_SIZE", "1024")
    monkeypatch.setenv("RAG_VECTOR_STORE__TOP_K", "12")
    monkeypatch.setenv("RAG_LLM__BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("RAG_ENVIRONMENT", "production")

    settings = Settings()

    assert settings.chunking.chunk_size == 1024
    assert settings.vector_store.top_k == 12
    assert settings.llm.base_url == "https://api.groq.com/openai/v1"
    assert settings.environment is Environment.PRODUCTION


def test_chunk_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    """The cross-field invariant is enforced through the environment too."""
    monkeypatch.setenv("RAG_CHUNKING__CHUNK_SIZE", "128")
    monkeypatch.setenv("RAG_CHUNKING__CHUNK_OVERLAP", "512")

    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings()


def test_misspelled_nested_variable_is_rejected(monkeypatch):
    """extra='forbid' on the sub-model turns a silent typo into a startup crash."""
    monkeypatch.setenv("RAG_CHUNKING__CHUNK_SIZ", "1024")

    with pytest.raises(ValidationError):
        Settings()


def test_secrets_are_masked_but_readable(monkeypatch):
    """SecretStr keeps the key out of logs, reprs and tracebacks."""
    monkeypatch.setenv("RAG_LLM__API_KEY", "sk-do-not-leak")

    settings = Settings()

    assert "sk-do-not-leak" not in str(settings)
    assert "sk-do-not-leak" not in repr(settings)
    assert settings.llm.api_key is not None
    assert settings.llm.api_key.get_secret_value() == "sk-do-not-leak"


def test_settings_are_immutable():
    """frozen=True must hold on the root model and on the sub-models alike."""
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.app_name = "other"

    with pytest.raises(ValidationError):
        settings.chunking.chunk_size = 128


def test_get_settings_returns_a_cached_instance():
    """get_settings is memoised: every caller shares the same object."""
    assert get_settings() is get_settings()
