from __future__ import annotations

from codex55_rag_project.data.vector_store.hybrid_pgvector import (
    HybridPgVectorStore,
    _build_hybrid_filter,
    _normalize_langchain_pg_connection,
    _normalize_psycopg_connection,
)


def test_hybrid_filter_always_includes_tenant_id() -> None:
    assert _build_hybrid_filter("tenant-a", {"source": "pdf"}) == {
        "tenant_id": "tenant-a",
        "source": "pdf",
    }


def test_jdbc_connection_is_normalized_for_langchain() -> None:
    print("测试")
    assert (
        _normalize_langchain_pg_connection("jdbc:postgresql://localhost:5432/rag_medical")
        == "postgresql+psycopg://localhost:5432/rag_medical"
    )


def test_sqlalchemy_connection_is_normalized_for_psycopg() -> None:
    assert (
        _normalize_psycopg_connection("postgresql+psycopg://localhost:5432/rag_medical")
        == "postgresql://localhost:5432/rag_medical"
    )


def test_hybrid_search_uses_pg_jieba_config() -> None:
    store = HybridPgVectorStore(
        connection="postgresql+psycopg://localhost:5432/rag_medical",
        table_name="medical_rag_chunks",
        embeddings=object(),
        embedding_length=1024,
    )

    config = store._new_hybrid_config("舌苔厚腻怎么办", 8)

    assert config.tsv_lang == "jiebacfg"
    assert config.tsv_column == "content_tsv"
    assert config.fts_query == "舌苔厚腻怎么办"
    assert config.primary_top_k == 8
    assert config.secondary_top_k == 8
