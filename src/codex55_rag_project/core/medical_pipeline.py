"""Production medical RAG pipeline with rewrite, hybrid multi-retrieval, rerank, and audit."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from codex55_rag_project.core.models import Answer, RetrievedChunk
from codex55_rag_project.core.ports import LLM, PromptBuilder, Reranker
from codex55_rag_project.data.audit import MedicalAuditStore, MedicalQueryAuditRecord
from codex55_rag_project.services.llm.query_rewriter import QueryRewriter, rewrite_with_fallback
from codex55_rag_project.services.retriever.multi_query import MultiQueryRetriever


@dataclass(frozen=True)
class MedicalRetrievalConfig:
    candidate_k: int = 24
    final_k: int = 6
    min_score: float = 0.0
    rewrite_count: int = 3


class MedicalRagPipeline:
    """Medical RAG orchestration with detailed audit logging."""

    def __init__(
        self,
        query_rewriter: QueryRewriter,
        retriever: MultiQueryRetriever,
        reranker: Reranker,
        prompt_builder: PromptBuilder,
        llm: LLM,
        audit_store: MedicalAuditStore,
        config: MedicalRetrievalConfig | None = None,
    ) -> None:
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.audit_store = audit_store
        self.config = config or MedicalRetrievalConfig()

    def ask(
        self,
        question: str,
        request_id: str,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> Answer:
        start = time.perf_counter()
        rewritten_queries = rewrite_with_fallback(self.query_rewriter, question, self.config.rewrite_count)
        candidates: list[RetrievedChunk] = []
        contexts: list[RetrievedChunk] = []
        answer_text = ""
        error: str | None = None
        try:
            candidates = self.retriever.retrieve_many(
                queries=rewritten_queries,
                top_k=self.config.candidate_k,
                tenant_id=tenant_id,
                metadata_filter=metadata_filter,
            )
            candidates = [item for item in candidates if item.score >= self.config.min_score]
            if not candidates:
                answer_text = "未在知识库中找到足够依据，无法从资料中确认。"
                return Answer(
                    question=question,
                    text=answer_text,
                    contexts=[],
                    metadata=self._metadata(request_id, rewritten_queries, 0, 0),
                )
            contexts = self.reranker.rerank(question, candidates, top_k=self.config.final_k)
            prompt = self.prompt_builder.build(question, contexts)
            answer = self.llm.generate(prompt=prompt, question=question, contexts=contexts)
            answer_text = answer.text
            return Answer(
                question=answer.question,
                text=answer.text,
                contexts=answer.contexts,
                metadata={
                    **answer.metadata,
                    **self._metadata(request_id, rewritten_queries, len(candidates), len(contexts)),
                },
            )
        except Exception as exc:
            error = str(exc)
            answer_text = ""
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.audit_store.write_query(
                MedicalQueryAuditRecord(
                    request_id=request_id,
                    tenant_id=tenant_id,
                    question=question,
                    rewritten_queries=rewritten_queries,
                    retrieved_chunks=_audit_chunks(candidates),
                    reranked_chunks=_audit_chunks(contexts),
                    answer=answer_text,
                    duration_ms=duration_ms,
                    error=error,
                )
            )

    @staticmethod
    def _metadata(
        request_id: str,
        rewritten_queries: list[str],
        candidate_count: int,
        final_count: int,
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "rewritten_queries": rewritten_queries,
            "retrieval_mode": "hybrid_multi_query",
            "candidate_count": candidate_count,
            "final_count": final_count,
        }


def _audit_chunks(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item.chunk.id,
            "document_id": item.chunk.document_id,
            "score": item.score,
            "source": item.chunk.metadata.get("source"),
            "page_number": item.chunk.metadata.get("page_number"),
            "chunk_index": item.chunk.metadata.get("chunk_index"),
        }
        for item in chunks
    ]
