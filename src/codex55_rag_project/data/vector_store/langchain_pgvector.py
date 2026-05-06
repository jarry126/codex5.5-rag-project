"""LangChain PGVector 向量存储适配器。

中文：生产链路使用 LangChain 官方 PGVector 集成，表结构由 langchain-postgres 管理。
English: Production path uses LangChain's PGVector integration; langchain-postgres manages tables.
"""

from __future__ import annotations

from typing import Any

from codex55_rag_project.core.models import Chunk, RetrievedChunk


class LangChainPgVectorStore:
    """LangChain PGVector adapter used by the production path.

    中文：生产链路使用 LangChain 官方 PGVector 集成，表结构和 collection 由 langchain-postgres 管理。
    English: The production path uses LangChain's PGVector integration; langchain-postgres manages tables
    and collections.
    """

    def __init__(
        self,
        connection: str,
        collection_name: str,
        embeddings: Any,
        embedding_length: int,
        pool_size: int = 10,
    ) -> None:
        # 延迟导入 LangChain，保证无外部依赖的单元测试仍能 import 项目基础模块。
        from langchain_postgres import PGVector

        self.connection = _normalize_langchain_pg_connection(connection)
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.embedding_length = embedding_length
        self.pool_size = pool_size
        # 初始化 LangChain PGVector 实例，embedding 会自动在 add_documents 时调用。
        self._store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=self.connection,
            use_jsonb=True,
            embedding_length=embedding_length,
        )

    def init_schema(self) -> None:
        """初始化数据库 schema。

        中文：显式创建 vector extension，避免数据库缺少扩展时首次请求才失败。
        English: Explicitly creates vector extension to avoid first-request failure when extension is missing.
        """
        # LangChain PGVector 会在初始化和首次写入时创建 collection 相关表。
        # 这里显式创建 vector extension，避免数据库缺少扩展时首次请求才失败。
        import psycopg

        with psycopg.connect(_normalize_psycopg_connection(self.connection)) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

    def health_check(self) -> None:
        """健康检查。

        中文：执行简单 SQL 查询确认数据库连接正常。
        English: Executes simple SQL query to confirm database connection is healthy.
        """
        import psycopg

        with psycopg.connect(_normalize_psycopg_connection(self.connection)) as conn:
            conn.execute("SELECT 1").fetchone()

    def close(self) -> None:
        """关闭连接池。

        中文：当前 langchain-postgres 实例没有暴露必须手动关闭的连接池；保留方法让 AppState 生命周期统一。
        English: langchain-postgres doesn't expose a connection pool to close; method kept for AppState lifecycle.
        """
        # 当前 langchain-postgres 实例没有暴露必须手动关闭的连接池；保留方法让 AppState 生命周期统一。
        return None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """将 chunk 写入向量库。

        中文：先删除旧 id 再写入，保证幂等；LangChain 在 add_documents 时自动调用 embedding。
        English: Delete old ids before writing for idempotency; LangChain calls embedding during add_documents.
        """
        if not chunks:
            return
        # LangChain Document 的 metadata 会进入 JSONB 字段；tenant_id 必须保留在 metadata 中用于 filter。
        documents = [_to_langchain_document(chunk) for chunk in chunks]
        ids = [chunk.id for chunk in chunks]
        # LangChain PGVector 的 add_documents 更偏追加语义；生产同步索引需要幂等，所以先删旧 id 再写入。
        self._store.delete(ids=ids)
        self._store.add_documents(documents, ids=ids)

    def similarity_search(
        self,
        query: str,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """相似度搜索。

        中文：将 tenant_id 和 metadata_filter 下推到 LangChain PGVector 的 JSONB filter。
        English: Pushes tenant_id and metadata_filter to LangChain PGVector's JSONB filter.
        """
        filter_ = _build_langchain_filter(tenant_id, metadata_filter)
        results = self._store.similarity_search_with_relevance_scores(query, k=top_k, filter=filter_)
        return [_from_langchain_result(document, score) for document, score in results]


def _to_langchain_document(chunk: Chunk) -> Any:
    """将内部 Chunk 转换为 LangChain Document。

    中文：metadata 中追加 chunk_id 和 document_id，便于搜索结果溯源。
    English: Adds chunk_id and document_id to metadata for result tracing.
    """
    from langchain_core.documents import Document as LangChainDocument

    metadata = {
        **chunk.metadata,
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
    }
    return LangChainDocument(page_content=chunk.text, metadata=metadata)


def _from_langchain_result(document: Any, score: float) -> RetrievedChunk:
    """将 LangChain 搜索结果转换为内部 RetrievedChunk。

    中文：从 metadata 中提取 chunk_id、document_id 和其他信息。
    English: Extracts chunk_id, document_id and other info from metadata.
    """
    metadata = dict(document.metadata)
    chunk_id = str(metadata.get("chunk_id", ""))
    document_id = str(metadata.get("document_id", ""))
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id=document_id,
            text=document.page_content,
            metadata=metadata,
        ),
        score=float(score),
    )


def _build_langchain_filter(tenant_id: str, metadata_filter: dict[str, Any] | None) -> dict[str, Any]:
    """构建 LangChain filter。

    中文：langchain-postgres 会把 filter 下推到 JSONB 查询；tenant_id 永远强制参与过滤。
    English: langchain-postgres pushes filter to JSONB query; tenant_id is always enforced.
    """
    # langchain-postgres 会把 filter 下推到 JSONB 查询；tenant_id 永远强制参与过滤。
    merged = {"tenant_id": tenant_id}
    if metadata_filter:
        merged.update(metadata_filter)
    return merged


def _normalize_langchain_pg_connection(connection: str) -> str:
    """规范化 LangChain PGVector 连接串。

    中文：langchain-postgres 文档推荐 postgresql+psycopg scheme。
    English: langchain-postgres docs recommend postgresql+psycopg scheme.
    """
    # langchain-postgres 文档推荐 postgresql+psycopg scheme。
    if connection.startswith("postgresql://"):
        return "postgresql+psycopg://" + connection.removeprefix("postgresql://")
    return connection


def _normalize_psycopg_connection(connection: str) -> str:
    """规范化 psycopg 连接串。

    中文：psycopg 自己不认识 SQLAlchemy 风格的 postgresql+psycopg scheme。
    English: psycopg doesn't recognize SQLAlchemy-style postgresql+psycopg scheme.
    """
    # psycopg 自己不认识 SQLAlchemy 风格的 postgresql+psycopg scheme。
    return connection.replace("postgresql+psycopg://", "postgresql://", 1)
