"""Multi-query retrieval and reciprocal-rank fusion."""

from __future__ import annotations

from typing import Any

from codex55_rag_project.core.models import RetrievedChunk
from codex55_rag_project.core.ports import Retriever

"""
    多问题召回
"""

class MultiQueryRetriever:
    """Runs several query variants and fuses the retrieved chunks."""

    def __init__(self, retriever: Retriever, rrf_k: int = 60) -> None:
        self.retriever = retriever
        self.rrf_k = rrf_k

    def retrieve_many(
        self,
        queries: list[str],
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        fused: dict[str, tuple[RetrievedChunk, float]] = {}
        for query in queries:
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                tenant_id=tenant_id,
                metadata_filter=metadata_filter,
            )
            for rank, item in enumerate(results, start=1):
                chunk_id = item.chunk.id
                rrf_score = 1.0 / (self.rrf_k + rank)
                combined_score = item.score + rrf_score
                current = fused.get(chunk_id)
                if current is None:
                    fused[chunk_id] = (item, combined_score)
                    continue
                existing, score = current
                if item.score > existing.score:
                    existing = item
                fused[chunk_id] = (existing, score + combined_score)

        ranked = [
            RetrievedChunk(chunk=item.chunk, score=score)
            for item, score in (value for value in fused.values())
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]
