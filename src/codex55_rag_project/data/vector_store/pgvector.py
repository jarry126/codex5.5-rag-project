from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

from codex55_rag_project.core.ports import Vector
from codex55_rag_project.core.models import Chunk, RetrievedChunk


VALID_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class PgVectorStore:
    """Postgres + pgvector implementation.

    中文：适合第一版生产，因为业务元数据、租户隔离和向量索引可以放在同一个事务型数据库中。
    English: A good v1 production choice because metadata, tenant isolation, and vector indexes live in one
    transactional database.
    """

    def __init__(self, dsn: str, table: str, dimensions: int, pool_size: int = 10) -> None:
        if not VALID_IDENTIFIER.match(table):
            raise ValueError("table must be a safe SQL identifier")
        # 中文：延迟导入 psycopg，避免本地纯单元测试或文档工具在没有数据库依赖时 import 失败。
        # English: Import psycopg lazily so pure unit tests and docs can import the package without DB deps.
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.table = table
        self.dimensions = dimensions
        self.pool: ConnectionPool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=pool_size,
            kwargs={"row_factory": dict_row},
        )

    def init_schema(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # 中文：tenant_id 是强隔离字段；metadata 用 JSONB 存权限、来源、分类等可过滤信息。
            # English: tenant_id is the hard isolation key; metadata stores filterable authorization/source fields.
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding VECTOR({self.dimensions}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_tenant_idx ON {self.table} (tenant_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_metadata_idx ON {self.table} USING GIN (metadata)")
            # 中文：ivfflat 适合中大型数据集；生产上应在数据量上来后 ANALYZE，并按规模调整 lists/probes。
            # English: ivfflat fits medium/large datasets; run ANALYZE and tune lists/probes as data grows.
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx
                ON {self.table}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            conn.commit()

    def health_check(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("SELECT 1").fetchone()

    def close(self) -> None:
        self.pool.close()

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        # 将业务对象转换成数据库行；tenant_id 从 metadata 中取出，作为独立字段便于强过滤和建索引。
        rows = [
            (
                chunk.id,
                chunk.document_id,
                str(chunk.metadata.get("tenant_id", "default")),
                chunk.text,
                json.dumps(chunk.metadata, ensure_ascii=False),
                _vector_literal(vector),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    f"""
                    INSERT INTO {self.table}
                        (chunk_id, document_id, tenant_id, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        tenant_id = EXCLUDED.tenant_id,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    # ON CONFLICT 保证同一个 chunk_id 重复索引时更新旧记录，而不是插入重复数据。
                    rows,
                )
            conn.commit()

    def search(
        self,
        query_vector: Vector,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        # 中文：过滤条件在数据库层执行，避免先召回再过滤导致越权片段进入候选集。
        # English: Push filters into SQL so unauthorized chunks never enter the candidate set.
        where, params = _build_search_filter_params(query_vector, tenant_id, metadata_filter, top_k)

        with self.pool.connection() as conn:
            return _search(conn, self.table, where, params)


def _build_search_filter_params(
    query_vector: Vector,
    tenant_id: str,
    metadata_filter: dict[str, Any] | None,
    top_k: int,
) -> tuple[str, list[Any]]:
    # 参数顺序要和 _build_search_sql 中的占位符保持一致：
    # query_vector 用于 score 和 order by，tenant/filter 用于 WHERE，top_k 用于 LIMIT。
    params: list[Any] = [_vector_literal(query_vector), tenant_id]
    where = "tenant_id = %s"
    if metadata_filter:
        # JSONB @> 表示 metadata 至少包含传入过滤条件；适合做来源、分类、权限标签过滤。
        where += " AND metadata @> %s::jsonb"
        params.append(json.dumps(metadata_filter, ensure_ascii=False))
    params.append(top_k)
    return where, params


def _build_search_sql(table: str, where: str) -> str:
    # pgvector 的 <=> 是 cosine distance；这里用 1 - distance 转成越大越好的相似度分数。
    return f"""
        SELECT
            chunk_id,
            document_id,
            text,
            metadata,
            1 - (embedding <=> %s::vector) AS score
        FROM {table}
        WHERE {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """


def _search(conn: Any, table: str, where: str, params: list[Any]) -> list[RetrievedChunk]:
    # 同一个 query_vector 在 SQL 中出现两次：一次算 score，一次排序。
    rows = conn.execute(_build_search_sql(table, where), [params[0], *params[1:-1], params[0], params[-1]]).fetchall()
    return [
        RetrievedChunk(
            chunk=Chunk(
                id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["text"],
                metadata=row["metadata"],
            ),
            score=float(row["score"]),
        )
        for row in rows
    ]


def _vector_literal(vector: Vector) -> str:
    # psycopg 传 pgvector 时使用字符串字面量最简单；统一保留 8 位小数，避免 SQL 太长。
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"
