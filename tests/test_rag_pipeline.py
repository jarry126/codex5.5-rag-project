from __future__ import annotations

from codex55_rag_project.bootstrap.local_factory import build_local_rag
from codex55_rag_project.core.models import Document


def test_rag_pipeline_returns_contexts() -> None:
    rag = build_local_rag(
        [
            Document(
                id="doc-1",
                text="RAG 需要通过接口隔离 Embedding、VectorStore、Retriever、Reranker 和 LLM。",
                metadata={"source": "unit-test"},
            )
        ]
    )

    answer = rag.ask("RAG 应该隔离哪些组件？")

    assert "Embedding" in answer.text
    assert answer.contexts
    assert answer.contexts[0].chunk.metadata["source"] == "unit-test"

