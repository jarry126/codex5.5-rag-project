"""LangChain PGVectorStore hybrid-search adapter."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Coroutine
from typing import Any, cast

from codex55_rag_project.core.models import Chunk, RetrievedChunk


class HybridPgVectorStore:
    """Adapter around langchain-postgres PGVectorStore with hybrid search enabled."""

    def __init__(
        self,
        connection: str,
        table_name: str,
        embeddings: Any,
        embedding_length: int,
        text_search_config: str = "jiebacfg",
    ) -> None:
        self.connection = _normalize_langchain_pg_connection(connection)
        self.table_name = table_name
        self.hybrid_index_name = f"{table_name}_tsv_idx"
        self.embeddings = embeddings
        self.embedding_length = embedding_length
        self.text_search_config = text_search_config
        self._engine: Any | None = None
        self._store: Any | None = None

    def init_schema(self) -> None:
        self._ensure_text_search_config()
        store = self._ensure_store()
        self._ensure_tsv_trigger()
        if hasattr(store, "apply_hybrid_search_index") and not self._hybrid_index_exists():
            store.apply_hybrid_search_index()

    def health_check(self) -> None:
        import psycopg

        with psycopg.connect(_normalize_psycopg_connection(self.connection)) as conn:
            conn.execute("SELECT 1").fetchone()

    def close(self) -> None:
        close = getattr(self._engine, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                coroutine = cast(Coroutine[Any, Any, Any], result)
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(coroutine)
                else:
                    loop.create_task(coroutine)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        store = self._ensure_store()
        documents = [_to_langchain_document(chunk) for chunk in chunks]
        ids = [chunk.id for chunk in chunks]
        delete = getattr(store, "delete", None)
        if callable(delete):
            delete(ids=ids)
        store.add_documents(documents=documents, ids=ids)

    def similarity_search(
        self,
        query: str,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        store = self._ensure_store()
        filter_ = _build_hybrid_filter(tenant_id, metadata_filter)
        results = _similarity_search_with_scores(
            store,
            query,
            top_k,
            filter_,
            self._new_hybrid_config(query, top_k),
        )
        return [_from_langchain_result(document, score) for document, score in results]

    def _new_hybrid_config(self, fts_query: str = "", top_k: int | None = None) -> Any:
        """创建一次混合检索配置。

        中文：HybridSearchConfig 不会在 Python 里生成关键词列表；它只是告诉
        langchain-postgres：向量检索之外，还要把 fts_query 交给 PostgreSQL，
        用 jiebacfg 对问题做全文检索，并用 reciprocal_rank_fusion 融合两路结果。
        """
        from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion


        """
            HybridSearchConfig 配置：
              关键词检索字段
              关键词分词配置
              关键词查询文本
              向量/关键词各取多少
              两路结果如何融合
        """
        config = HybridSearchConfig(
            # content_tsv 是数据库里由 trigger 自动生成的 tsvector 字段，用于关键词/全文检索。
            tsv_column="content_tsv",
            # text_search_config 当前默认是 jiebacfg，也就是 pg_jieba 提供的中文分词配置。
            tsv_lang=self.text_search_config,
            # fts_query 是用户问题或改写后的问题；真正的分词和 tsquery 构造发生在 PostgreSQL。
            fts_query=fts_query,
            # RRF 用来融合向量召回排名和全文检索排名，避免只相信其中一路结果。
            fusion_function=reciprocal_rank_fusion,
            # GIN 索引名称；apply_hybrid_search_index 会使用它创建 content_tsv 索引。
            index_name=self.hybrid_index_name,
        )
        if top_k is not None:
            # primary_top_k 控制向量检索候选数，secondary_top_k 控制全文检索候选数。
            config.primary_top_k = top_k
            config.secondary_top_k = top_k
        return config

    def _ensure_store(self) -> Any:
        if self._store is not None:
            return self._store

        from langchain_postgres import PGEngine, PGVectorStore
        from langchain_postgres.v2.engine import Column

        self._engine = PGEngine.from_connection_string(url=self.connection)
        hybrid_config = self._new_hybrid_config()
        init_table = getattr(self._engine, "init_vectorstore_table", None)
        if callable(init_table) and not self._table_exists():
            try:
                init_table(
                    table_name=self.table_name,
                    vector_size=self.embedding_length,
                    id_column=Column("langchain_id", "TEXT", nullable=False),
                    hybrid_search_config=hybrid_config,
                )
            except TypeError:
                init_table(
                    self.table_name,
                    self.embedding_length,
                    id_column=Column("langchain_id", "TEXT", nullable=False),
                    hybrid_search_config=hybrid_config,
                )
        self._store = PGVectorStore.create_sync(
            engine=self._engine,
            table_name=self.table_name,
            embedding_service=self.embeddings,
            hybrid_search_config=hybrid_config,
        )
        return self._store

    def _ensure_text_search_config(self) -> None:
        import psycopg

        with psycopg.connect(_normalize_psycopg_connection(self.connection)) as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM pg_ts_config
                WHERE cfgname = %s
                """,
                (self.text_search_config.split(".")[-1],),
            ).fetchone()
        if not row or row[0] is None:
            raise RuntimeError(
                f"PostgreSQL text search config '{self.text_search_config}' does not exist. "
                "Run CREATE EXTENSION IF NOT EXISTS pg_jieba in the medical database first."
            )

    def _table_exists(self) -> bool:
        import psycopg

        with psycopg.connect(_normalize_psycopg_connection(self.connection)) as conn:
            row = conn.execute("SELECT to_regclass(%s)", (f"public.{self.table_name}",)).fetchone()
        return bool(row and row[0])

    def _ensure_tsv_trigger(self) -> None:
        import psycopg

        function_name = f"{self.table_name}_content_tsv_update"
        trigger_name = f"{self.table_name}_content_tsv_trigger"
        with psycopg.connect(_normalize_psycopg_connection(self.connection)) as conn:
            conn.execute(
                f"""
                CREATE OR REPLACE FUNCTION {function_name}()
                RETURNS trigger AS $$
                BEGIN
                    NEW.content_tsv := to_tsvector(
                        '{self.text_search_config}',
                        COALESCE(NEW.content, '')
                    );
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """
            )
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON {self.table_name}")
            conn.execute(
                f"""
                CREATE TRIGGER {trigger_name}
                BEFORE INSERT OR UPDATE OF content
                ON {self.table_name}
                FOR EACH ROW
                EXECUTE FUNCTION {function_name}()
                """
            )
            conn.commit()

    def _hybrid_index_exists(self) -> bool:
        import psycopg

        with psycopg.connect(_normalize_psycopg_connection(self.connection)) as conn:
            row = conn.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND tablename = %s
                      AND indexname IN (%s, %s)
                )
                """,
                (self.table_name, self.hybrid_index_name, "langchain_tsv_index"),
            ).fetchone()
        return bool(row and row[0])


def _similarity_search_with_scores(
    store: Any,
    query: str,
    top_k: int,
    filter_: dict[str, Any],
    hybrid_search_config: Any | None = None,
) -> list[tuple[Any, float]]:
    """执行 LangChain PGVectorStore 检索，并尽量返回分数。

    中文：不同版本的 langchain-postgres 暴露的方法名不完全一致，所以这里按能力从
    relevance_scores -> score -> documents 逐级降级。传入 hybrid_search_config 时，
    LangChain 会同时执行向量检索和 PostgreSQL 全文检索，然后融合结果。
    """
    # hybrid_search_config 包含 content_tsv、jiebacfg 和 fts_query；有它才会触发混合检索。
    kwargs = {"hybrid_search_config": hybrid_search_config} if hybrid_search_config is not None else {}

    # 优先使用 relevance score 版本：返回 (Document, relevance_score)，分数通常越大越相关。
    if hasattr(store, "similarity_search_with_relevance_scores"):
        try:
            return list(store.similarity_search_with_relevance_scores(query=query, k=top_k, filter=filter_, **kwargs))
        except TypeError:
            # 兼容旧版本签名：有些版本不接受 query= 这种关键字参数。
            return list(store.similarity_search_with_relevance_scores(query, k=top_k, filter=filter_, **kwargs))

    # 次选 score 版本：也返回 (Document, score)，但不同库版本里 score 语义可能是距离或相关性。
    if hasattr(store, "similarity_search_with_score"):
        try:
            return list(store.similarity_search_with_score(query=query, k=top_k, filter=filter_, **kwargs))
        except TypeError:
            # 兼容旧版本签名。
            return list(store.similarity_search_with_score(query, k=top_k, filter=filter_, **kwargs))

    # 最后降级为只返回 Document 的检索方法；没有分数时统一补 0.0，保证上层返回结构稳定。
    try:
        documents = store.similarity_search(query=query, k=top_k, filter=filter_, **kwargs)
    except TypeError:
        # 兼容旧版本签名。
        documents = store.similarity_search(query, k=top_k, filter=filter_, **kwargs)
    return [(document, 0.0) for document in documents]


def _to_langchain_document(chunk: Chunk) -> Any:
    from langchain_core.documents import Document as LangChainDocument

    metadata = {
        **chunk.metadata,
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
    }
    return LangChainDocument(page_content=chunk.text, metadata=metadata)


def _from_langchain_result(document: Any, score: float) -> RetrievedChunk:
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


def _build_hybrid_filter(tenant_id: str, metadata_filter: dict[str, Any] | None) -> dict[str, Any]:
    merged = {"tenant_id": tenant_id}
    if metadata_filter:
        merged.update(metadata_filter)
    return merged


def _normalize_langchain_pg_connection(connection: str) -> str:
    if connection.startswith("jdbc:postgresql://"):
        connection = "postgresql://" + connection.removeprefix("jdbc:postgresql://")
    if connection.startswith("postgresql://"):
        return "postgresql+psycopg://" + connection.removeprefix("postgresql://")
    return connection


def _normalize_psycopg_connection(connection: str) -> str:
    if connection.startswith("jdbc:postgresql://"):
        return "postgresql://" + connection.removeprefix("jdbc:postgresql://")
    return connection.replace("postgresql+psycopg://", "postgresql://", 1)
