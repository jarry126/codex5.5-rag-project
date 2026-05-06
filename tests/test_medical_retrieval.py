from __future__ import annotations

from codex55_rag_project.core.medical_pipeline import MedicalRagPipeline, MedicalRetrievalConfig
from codex55_rag_project.core.models import Answer, Chunk, RetrievedChunk
from codex55_rag_project.services.llm.query_rewriter import rewrite_with_fallback
from codex55_rag_project.services.retriever.multi_query import MultiQueryRetriever


def chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=chunk_id, document_id="doc", text=f"text {chunk_id}", metadata={"tenant_id": "tenant-a"}),
        score=score,
    )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object] | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, tenant_id, metadata_filter))
        if query == "q1":
            return [chunk("a", 0.8), chunk("b", 0.7)]
        return [chunk("b", 0.9), chunk("c", 0.6)]


class FailingRewriter:
    def rewrite(self, question: str, count: int) -> list[str]:
        raise RuntimeError("model unavailable")


def test_rewrite_with_fallback_returns_original_question() -> None:
    assert rewrite_with_fallback(FailingRewriter(), "原问题", 3) == ["原问题"]


def test_multi_query_retriever_deduplicates_and_forwards_filters() -> None:
    fake = FakeRetriever()
    retriever = MultiQueryRetriever(fake)

    results = retriever.retrieve_many(
        queries=["q1", "q2"],
        top_k=5,
        tenant_id="tenant-a",
        metadata_filter={"source": "pdf"},
    )

    assert [item.chunk.id for item in results] == ["b", "a", "c"]
    assert fake.calls == [
        ("q1", "tenant-a", {"source": "pdf"}),
        ("q2", "tenant-a", {"source": "pdf"}),
    ]


class EmptyMultiRetriever:
    def retrieve_many(
        self,
        queries: list[str],
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        return []


class FakeReranker:
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return candidates[:top_k]


class FakePromptBuilder:
    def build(self, question: str, contexts: list[RetrievedChunk]) -> str:
        return "prompt"


class FailingLLM:
    def generate(self, prompt: str, question: str, contexts: list[RetrievedChunk]) -> Answer:
        raise AssertionError("LLM should not be called when retrieval is empty")


class FakeAuditStore:
    def __init__(self) -> None:
        self.records: list[object] = []

    def write_query(self, record: object) -> None:
        self.records.append(record)


def test_medical_pipeline_returns_controlled_answer_on_empty_retrieval() -> None:
    audit = FakeAuditStore()
    pipeline = MedicalRagPipeline(
        query_rewriter=FailingRewriter(),
        retriever=EmptyMultiRetriever(),
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        llm=FailingLLM(),
        audit_store=audit,
        config=MedicalRetrievalConfig(candidate_k=5, final_k=2, rewrite_count=3),
    )

    answer = pipeline.ask(question="不存在的问题", request_id="req-1", tenant_id="tenant-a")

    assert answer.text == "未在知识库中找到足够依据，无法从资料中确认。"
    assert answer.contexts == []
    assert answer.metadata["rewritten_queries"] == ["不存在的问题"]
    assert answer.metadata["candidate_count"] == 0
    assert len(audit.records) == 1
