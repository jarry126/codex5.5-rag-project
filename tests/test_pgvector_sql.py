from __future__ import annotations

import json

from codex55_rag_project.data.vector_store.pgvector import _build_search_filter_params, _build_search_sql


def test_search_filter_params_include_tenant_and_metadata_filter() -> None:
    where, params = _build_search_filter_params(
        query_vector=[0.1, 0.2],
        tenant_id="tenant-a",
        metadata_filter={"category": "policy"},
        top_k=5,
    )

    assert where == "tenant_id = %s AND metadata @> %s::jsonb"
    assert params[0] == "[0.10000000,0.20000000]"
    assert params[1] == "tenant-a"
    assert json.loads(params[2]) == {"category": "policy"}
    assert params[3] == 5


def test_search_sql_pushes_filter_into_database() -> None:
    sql = _build_search_sql("rag_chunks", "tenant_id = %s AND metadata @> %s::jsonb")

    assert "WHERE tenant_id = %s AND metadata @> %s::jsonb" in sql
    assert "ORDER BY embedding <=> %s::vector" in sql

