"""向量检索器实现。

中文：将查询文本向量化后交给 VectorStore 搜索，支持租户隔离和 metadata 过滤。
English: Vectorizes query text and delegates to VectorStore, supporting tenant isolation and metadata filtering.
"""

from __future__ import annotations

from typing import Any, Protocol

from codex55_rag_project.core.ports import Embedder, VectorStore
from codex55_rag_project.core.models import RetrievedChunk


class VectorRetriever:
    """本地 demo 使用的基础向量检索器。

    中文：适合无租户隔离的本地测试场景。
    English: Suitable for local testing without tenant isolation.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """检索与查询最相关的 chunk。

        中文：查询文本先转向量，再交给 vector_store 做相似度搜索。
        English: Query text is vectorized first, then passed to vector_store for similarity search.
        """
        # 查询文本先转向量，再交给 vector_store 做相似度搜索。
        query_vector = self.embedder.embed_texts([query])[0]
        return self.vector_store.search(
            query_vector,
            top_k=top_k,
            tenant_id=tenant_id,
            metadata_filter=metadata_filter,
        )


class TenantAwareSearchStore(Protocol):
    """支持租户隔离和 metadata 过滤的向量库协议。

    中文：生产向量库必须实现此协议，确保 tenant_id 在数据库层过滤。
    English: Production vector stores must implement this protocol, ensuring tenant_id filtering at DB level.
    """

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Search with tenant isolation and optional metadata filters."""


class TenantAwareVectorRetriever:
    """生产查询入口使用的检索器。

    中文：它强制把 tenant_id 和 metadata_filter 传到向量库，避免先查全量再在应用层过滤。
    English: Forces tenant_id and metadata_filter to be pushed to vector store, avoiding post-retrieval filtering.
    """

    def __init__(self, embedder: Embedder, vector_store: TenantAwareSearchStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """检索时传递租户和权限过滤条件。

        中文：权限和租户过滤不在这里做，而是继续传给 vector_store，让数据库层过滤。
        English: Authorization and tenant filtering is delegated to vector_store, done at database level.
        """
        # 注意：权限和租户过滤不在这里做，而是继续传给 vector_store，让数据库层过滤。
        query_vector = self.embedder.embed_texts([query])[0]
        return self.vector_store.search(
            query_vector,
            top_k=top_k,
            tenant_id=tenant_id,
            metadata_filter=metadata_filter,
        )
