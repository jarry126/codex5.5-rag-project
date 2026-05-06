from __future__ import annotations

from codex55_rag_project.data.vector_store.langchain_pgvector import (
    _build_langchain_filter,
    _normalize_langchain_pg_connection,
    _normalize_psycopg_connection,
)


def test_langchain_filter_always_includes_tenant_id() -> None:
    filter_ = _build_langchain_filter("tenant-a", {"category": "policy"})

    assert filter_ == {"tenant_id": "tenant-a", "category": "policy"}


def test_postgres_connection_is_normalized_for_langchain_pgvector() -> None:
    assert (
        _normalize_langchain_pg_connection("postgresql://rag:rag@localhost:5432/rag")
        == "postgresql+psycopg://rag:rag@localhost:5432/rag"
    )


def test_langchain_connection_is_normalized_for_psycopg_health_check() -> None:
    assert (
        _normalize_psycopg_connection("postgresql+psycopg://rag:rag@localhost:5432/rag")
        == "postgresql://rag:rag@localhost:5432/rag"
    )

