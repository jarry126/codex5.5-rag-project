from __future__ import annotations

import pytest

from codex55_rag_project.api.dependencies import build_app_state
from codex55_rag_project.config.settings import Settings


MODEL_ENV_VARS = [
    "DASHSCOPE_API_KEY",
    "RAG_DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
    "RAG_OPENAI_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
]


def test_openai_compatible_base_url_is_configurable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for env_var in MODEL_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    settings = Settings(
        openai_api_key="test-key",
        openai_base_url="https://models.example.com/v1",
        _env_file=None,
    )

    assert settings.openai_base_url == "https://models.example.com/v1"
    assert settings.embedding_model == "text-embedding-v4"
    assert settings.embedding_dimensions == 1024
    assert settings.chat_model == "qwen-plus"


def test_app_state_requires_model_api_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for env_var in MODEL_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    settings = Settings(openai_api_key="", _env_file=None)

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY or OPENAI_API_KEY"):
        build_app_state(settings)
