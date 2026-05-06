from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from codex55_rag_project.api.app import app
from codex55_rag_project.api.dependencies import get_app_state
from codex55_rag_project.core.models import Answer, Chunk, RetrievedChunk


class FakeIngestion:
    def run(self) -> SimpleNamespace:
        return SimpleNamespace(documents=1, chunks=1)


class FakePipeline:
    def ask(self, question: str, tenant_id: str = "default", metadata_filter: dict[str, object] | None = None) -> Answer:
        return Answer(
            question=question,
            text="answer with citation",
            contexts=[
                RetrievedChunk(
                    chunk=Chunk(
                        id="chunk-1",
                        document_id="doc-1",
                        text="context text",
                        metadata={"source": "unit-test", "tenant_id": tenant_id},
                    ),
                    score=0.91,
                )
            ],
            metadata={"model": "fake"},
        )


class FakeMedicalPipeline:
    def ask(
        self,
        question: str,
        request_id: str,
        tenant_id: str = "default",
        metadata_filter: dict[str, object] | None = None,
    ) -> Answer:
        return Answer(
            question=question,
            text="medical answer",
            contexts=[
                RetrievedChunk(
                    chunk=Chunk(
                        id="medical-chunk-1",
                        document_id="medical-doc-1",
                        text="medical context",
                        metadata={"source": "medical-pdf", "tenant_id": tenant_id},
                    ),
                    score=0.88,
                )
            ],
            metadata={
                "request_id": request_id,
                "rewritten_queries": [question, "改写问题"],
                "retrieval_mode": "hybrid_multi_query",
                "candidate_count": 3,
                "final_count": 1,
            },
        )


class FakeAppState:
    pipeline = FakePipeline()
    medical_pipeline = FakeMedicalPipeline()

    def build_ingestion(self, loader: object) -> FakeIngestion:
        return FakeIngestion()

    def ingest_pdf(self, pdf_path: str | None, tenant_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            ingestion_id="ingestion-1",
            documents=2,
            pages=2,
            chunks=5,
            source=pdf_path or "中医临床诊疗智能助手.pdf",
        )

    def health_check(self) -> None:
        return None


"""
    monkeypatch 的作用是临时修改某些东西，测试结束后自动恢复。
"""

def test_healthz_does_not_require_api_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("codex55_rag_project.security.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))

    response = TestClient(app).get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_requires_api_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("codex55_rag_project.security.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))
    app.dependency_overrides[get_app_state] = lambda: FakeAppState()

    try:
        response = TestClient(app).post("/v1/query", json={"tenant_id": "tenant-a", "question": "hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401


def test_query_response_contains_answer_citations_and_metadata(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("codex55_rag_project.security.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))
    app.dependency_overrides[get_app_state] = lambda: FakeAppState()

    try:
        response = TestClient(app).post(
            "/v1/query",
            headers={"x-api-key": "secret"},
            json={"tenant_id": "tenant-a", "question": "hello", "metadata_filter": {"source": "unit-test"}},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answer with citation"
    assert body["metadata"] == {"model": "fake"}
    assert body["citations"][0]["chunk_id"] == "chunk-1"


def test_pdf_ingestion_requires_api_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("codex55_rag_project.security.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))
    app.dependency_overrides[get_app_state] = lambda: FakeAppState()

    try:
        response = TestClient(app).post("/v1/admin/pdf-ingestions", json={"tenant_id": "tenant-a"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401


def test_pdf_ingestion_response_contains_stats(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("codex55_rag_project.security.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))
    app.dependency_overrides[get_app_state] = lambda: FakeAppState()

    try:
        response = TestClient(app).post(
            "/v1/admin/pdf-ingestions",
            headers={"x-api-key": "secret"},
            json={"tenant_id": "tenant-a", "pdf_path": "中医临床诊疗智能助手.pdf"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["ingestion_id"] == "ingestion-1"
    assert body["pages"] == 2
    assert body["chunks"] == 5


def test_medical_query_response_contains_rewrite_metadata(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("codex55_rag_project.security.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))
    app.dependency_overrides[get_app_state] = lambda: FakeAppState()

    try:
        response = TestClient(app).post(
            "/v1/medical/query",
            headers={"x-api-key": "secret", "x-request-id": "req-1"},
            json={"tenant_id": "tenant-a", "question": "失眠怎么办"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "medical answer"
    assert body["metadata"]["request_id"] == "req-1"
    assert body["metadata"]["retrieval_mode"] == "hybrid_multi_query"
    assert body["metadata"]["candidate_count"] == 3
    assert body["citations"][0]["chunk_id"] == "medical-chunk-1"

def test_readyz_real_startup() -> None:
    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 200
