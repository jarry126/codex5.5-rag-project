"""内存向量存储（用于测试）。

中文：将向量保存在内存中，适合单元测试和本地调试，不支持持久化。
English: Stores vectors in memory, suitable for unit tests and local debugging, no persistence.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

from codex55_rag_project.core.ports import Vector
from codex55_rag_project.core.models import Chunk, RetrievedChunk


@dataclass(frozen=True)
class _VectorRecord:
    """内部向量记录。

    中文：存储 chunk 和对应的向量，用于内存检索。
    English: Stores chunk and its vector for in-memory retrieval.
    """

    chunk: Chunk
    vector: Vector


class InMemoryVectorStore:
    """内存向量存储。

    中文：将 chunk 和向量保存在内存 dict 中，使用 cosine similarity 搜索。
    English: Stores chunks and vectors in memory dict, uses cosine similarity for search.
    """

    def __init__(self) -> None:
        # 内部使用 dict 存储，key 是 chunk_id。
        self._records: dict[str, _VectorRecord] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        """写入向量。

        中文：chunk 和 vectors 数量必须一致，按 chunk.id 存入内部 dict。
        English: chunk and vectors counts must match; stored in internal dict keyed by chunk.id.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors):
            self._records[chunk.id] = _VectorRecord(chunk=chunk, vector=vector)

    def search(
        self,
        query_vector: Vector,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """相似度搜索。

        中文：先过滤 tenant_id 和 metadata，再计算 cosine similarity，返回 top_k。
        English: Filters by tenant_id and metadata, computes cosine similarity, returns top_k.
        """
        def matches(chunk: Chunk) -> bool:
            # 租户隔离：metadata 中的 tenant_id 必须匹配。
            if chunk.metadata.get("tenant_id", "default") != tenant_id:
                return False
            # metadata 过滤：所有 key-value 必须完全匹配。
            if not metadata_filter:
                return True
            return all(chunk.metadata.get(key) == value for key, value in metadata_filter.items())

        # 计算所有匹配记录的相似度，返回 top_k。
        scored = (
            RetrievedChunk(chunk=record.chunk, score=_cosine_similarity(query_vector, record.vector))
            for record in self._records.values()
            if matches(record.chunk)
        )
        return heapq.nlargest(top_k, scored, key=lambda item: item.score)


def _cosine_similarity(left: Vector, right: Vector) -> float:
    """计算 cosine similarity。

    中文：两个向量点积除以模长乘积；这里简化为点积（假设向量已归一化）。
    English: Dot product divided by magnitude product; simplified to dot product (assuming normalized vectors).
    """
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    return sum(a * b for a, b in zip(left, right))
