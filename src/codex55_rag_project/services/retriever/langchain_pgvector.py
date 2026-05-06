"""LangChain PGVector 检索器。

中文：将查询文本直接传给 LangChain PGVector，embedding 和搜索由 LangChain 处理。
English: Passes query text directly to LangChain PGVector; LangChain handles embedding and search.
"""

from __future__ import annotations

from typing import Any, Protocol

from codex55_rag_project.core.models import RetrievedChunk


class SimilaritySearchStore(Protocol):
    def similarity_search(
        self,
        query: str,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Search chunks by query text."""


class LangChainPgVectorRetriever:
    """Retriever backed by LangChain PGVector.

    中文：查询文本直接交给 LangChain PGVector；embedding、metadata filter 和相似度搜索由 LangChain 处理。
    English: Query text goes directly to LangChain PGVector; LangChain handles embedding, metadata filtering,
    and similarity search.
    """

    def __init__(self, vector_store: SimilaritySearchStore) -> None:
        # vector_store 是 LangChainPgVectorStore 实例，封装了 LangChain PGVector。
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """检索与查询最相关的 chunk。

        中文：tenant_id 和 metadata_filter 会下推到 LangChain PGVector 的 JSONB filter。
        English: tenant_id and metadata_filter are pushed to LangChain PGVector's JSONB filter.
        """
        return self.vector_store.similarity_search(
            query=query,
            top_k=top_k,
            tenant_id=tenant_id,
            metadata_filter=metadata_filter,
        )
