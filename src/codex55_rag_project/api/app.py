from __future__ import annotations

import uuid
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status

from codex55_rag_project.api.dependencies import (
    AppState,
    build_app_state,
    get_app_state,
    set_app_state,
)
from codex55_rag_project.api.schemas import (
    Citation,
    IngestRequest,
    IngestResponse,
    MedicalQueryRequest,
    PdfIngestionRequest,
    PdfIngestionResponse,
    QueryRequest,
    QueryResponse,
    RagasEvaluationItem,
    RagasEvaluationRequest,
    RagasEvaluationResponse,
)
from codex55_rag_project.config import get_settings
from codex55_rag_project.core.models import Document
from codex55_rag_project.loaders.text import StaticDocumentLoader
from codex55_rag_project.monitoring import clear_request_id, configure_logging, get_logger, set_request_id
from codex55_rag_project.security import verify_api_key
from codex55_rag_project.services.evaluation.ragas_medical import (
    MedicalRagasCase,
    evaluate_medical_rag_cases,
)

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # FastAPI 启动时只初始化一次完整 RAG 链路，后续请求都复用 AppState。
    logger.info("Initializing RAG service")
    state = build_app_state(settings)
    state.init_schema()
    set_app_state(state)
    logger.info("RAG service initialized")
    try:
        yield
    finally:
        # 应用退出时统一释放外部资源；当前主要是 pgvector 连接池。
        logger.info("Closing RAG service")
        state.close()
        set_app_state(None)


app = FastAPI(title=settings.service_name, version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next):  # type: ignore[no-untyped-def]
    # 中文：request id 会回写到响应头，便于把客户端报错、网关日志和服务日志串起来。
    # English: Echo request id in the response so client, gateway, and service logs can be correlated.
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    status_code = 500
    set_request_id(request_id)
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "http_request",
            extra={
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )
        clear_request_id()


@app.get("/healthz", summary="存活检查")
def healthz() -> dict[str, str]:
    # 中文：存活检查不访问外部依赖，适合容器进程探活。
    # English: Liveness avoids external dependencies and is safe for container process probes.
    return {"status": "ok"}


@app.get("/readyz", summary="就绪检查")
def readyz(state: AppState = Depends(get_app_state)) -> dict[str, str]:
    # 中文：就绪检查会访问数据库，适合发布系统判断实例能否接流量。
    # English: Readiness checks the database and tells rollout systems whether the instance can serve traffic.
    state.health_check()
    return {"status": "ready"}


@app.post(
    "/v1/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_api_key)],
    summary="通用文本文档入库",
    description=(
        "同步切分、向量化并持久化调用方提交的文本文档，写入通用 RAG 向量库。"
        "接口受 `x-api-key` 保护，并且会把 `tenant_id` 写入 metadata，用于多租户隔离。"
    ),
    responses={401: {"description": "API Key 缺失或无效"}},
)
def ingest(
    request: IngestRequest,
    state: AppState = Depends(get_app_state),
) -> IngestResponse:
    """将原始文本文档索引到通用向量库。"""
    # 中文：第一版采用同步索引，请求返回时文档已经完成切分、向量化和入库。
    # English: v1 uses synchronous indexing; when the request returns, chunks are embedded and persisted.
    documents = [
        Document(
            id=document.id,
            text=document.text,
            # tenant_id 写入 metadata，后续 chunk 和向量入库时会一起保存，用于多租户隔离。
            metadata={**document.metadata, "tenant_id": request.tenant_id},
        )
        for document in request.documents
    ]
    stats = state.build_ingestion(StaticDocumentLoader(documents)).run()
    return IngestResponse(documents=stats.documents, chunks=stats.chunks)

@app.post(
    "/v1/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],
    summary="通用 RAG 问答",
    description=(
        "基于通用向量库执行原始 RAG 问答流程。"
        "`tenant_id` 和 `metadata_filter` 会下推到检索层，避免跨租户数据泄漏。"
    ),
    responses={401: {"description": "API Key 缺失或无效"}},
)
def query(
    request: QueryRequest, state: AppState = Depends(get_app_state)
) -> QueryResponse:
    """使用通用 RAG pipeline 回答用户问题。"""
    # 中文：tenant_id 和 metadata_filter 会下推到 pgvector 查询，避免应用层事后过滤带来的泄漏风险。
    # English: tenant_id and metadata_filter are pushed down to pgvector instead of filtered after retrieval.
    answer = state.pipeline.ask(
        question=request.question,
        tenant_id=request.tenant_id,
        metadata_filter=request.metadata_filter,
    )
    # API 层只负责把内部 Answer 转成 HTTP 响应；回答生成、引用选择都在 pipeline 内完成。
    return QueryResponse(
        answer=answer.text,
        citations=[
            Citation(
                chunk_id=item.chunk.id,
                document_id=item.chunk.document_id,
                source=item.chunk.metadata.get("source"),
                score=item.score,
                text=item.chunk.text,
                metadata=item.chunk.metadata,
            )
            for item in answer.contexts
        ],
        metadata=answer.metadata,
    )


@app.post(
    "/v1/admin/pdf-ingestions",
    response_model=PdfIngestionResponse,
    dependencies=[Depends(verify_api_key)],
    summary="医疗 PDF 入库",
    description=(
        "管理接口，用于把服务端本地 PDF 索引到医疗 RAG 知识库。"
        "流程会逐页提取文本，保留页码和来源 metadata，切分文本，调用 DashScope embedding，"
        "并将向量和混合检索所需字段写入 `rag_medical`。"
        "如果不传 `pdf_path`，则使用配置中的默认 PDF 路径。"
    ),
    responses={
        401: {"description": "API Key 缺失或无效"},
        502: {"description": "PDF 解析、embedding 服务或向量库写入失败"},
    },
)
def ingest_pdf(
    request: PdfIngestionRequest,
    state: AppState = Depends(get_app_state),
) -> PdfIngestionResponse:
    """将本地 PDF 索引到医疗混合检索向量库。

    这个接口是管理接口：它读取服务端本地文件路径，而不是接收任意上传文件。
    生产环境应只暴露给可信运维人员或内部自动化任务。
    """
    try:
        # AppState 持有入库 pipeline，路由层不重复创建模型客户端和数据库引擎。
        stats = state.ingest_pdf(pdf_path=request.pdf_path, tenant_id=request.tenant_id)
    except Exception as exc:
        # HTTP 响应保持简洁安全；结构化日志保留 traceback，便于运维排查。
        logger.exception(
            "pdf_ingestion_failed",
            extra={"event": "pdf_ingestion_failed", "tenant_id": request.tenant_id},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="PDF 入库失败，请检查模型服务凭证和服务日志",
        ) from exc
    logger.info(
        "pdf_ingestion_completed",
        extra={
            "event": "pdf_ingestion_completed",
            "tenant_id": request.tenant_id,
            "ingestion_id": stats.ingestion_id,
            "documents": stats.documents,
            "pages": stats.pages,
            "chunks": stats.chunks,
        },
    )
    return PdfIngestionResponse(
        ingestion_id=stats.ingestion_id,
        documents=stats.documents,
        pages=stats.pages,
        chunks=stats.chunks,
        source=stats.source,
    )


@app.post(
    "/v1/medical/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],
    summary="医疗混合检索 RAG 问答",
    description=(
        "面向客户的医疗 RAG 问答接口。"
        "接口会将原问题改写成多个检索问题，执行向量 + 关键词混合检索，融合并重排序候选片段，"
        "再调用 LLM 生成最终答案，返回引用片段，并写入审计记录以便追溯。"
    ),
    responses={
        401: {"description": "API Key 缺失或无效"},
        502: {"description": "问题改写、检索、重排序、LLM 生成或审计写入失败"},
    },
)
def medical_query(
    request: MedicalQueryRequest,
    http_request: Request,
    state: AppState = Depends(get_app_state),
) -> QueryResponse:
    """使用医疗多问题混合检索 RAG pipeline 回答客户问题。"""
    request_id = http_request.headers.get("x-request-id", str(uuid.uuid4()))
    try:
        # 医疗 pipeline 负责：问题改写 -> 多路召回 -> 融合 -> 重排序 -> Prompt -> LLM -> 审计记录。
        answer = state.medical_pipeline.ask(
            question=request.question,
            request_id=request_id,
            tenant_id=request.tenant_id,
            metadata_filter=request.metadata_filter,
        )
    except Exception as exc:
        # pipeline 内部会尽量记录审计；这里负责保护对外 API 形态，避免直接暴露内部异常。
        logger.exception(
            "medical_query_failed",
            extra={"event": "medical_query_failed", "request_id": request_id, "tenant_id": request.tenant_id},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="医疗问答失败，请检查模型服务凭证和服务日志",
        ) from exc
    logger.info(
        "medical_query_completed",
        extra={
            "event": "medical_query_completed",
            "request_id": request_id,
            "tenant_id": request.tenant_id,
            "query_count": len(answer.metadata.get("rewritten_queries", [])),
            "candidate_count": answer.metadata.get("candidate_count"),
            "final_count": answer.metadata.get("final_count"),
        },
    )
    return QueryResponse(
        answer=answer.text,
        citations=[
            Citation(
                chunk_id=item.chunk.id,
                document_id=item.chunk.document_id,
                source=item.chunk.metadata.get("source"),
                score=item.score,
                text=item.chunk.text,
                metadata=item.chunk.metadata,
            )
            for item in answer.contexts
        ],
        metadata=answer.metadata,
    )


@app.post(
    "/v1/admin/ragas-evaluations",
    response_model=RagasEvaluationResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Ragas 医疗 RAG 评估",
    description=(
        "管理接口，用于直接通过 HTTP 触发 Ragas 评估。"
        "每条用例都会先走真实医疗 RAG pipeline，从 Postgres/pgvector 检索上下文并生成答案，"
        "然后用 Ragas 指标评估回答是否有上下文支撑以及是否符合评分说明。"
        "该接口会调用真实大模型，建议只在本地、测试环境或内部管理网络使用。"
    ),
    responses={
        401: {"description": "API Key 缺失或无效"},
        502: {"description": "RAG 查询或 Ragas 评分失败"},
    },
)
def evaluate_ragas(
    request: RagasEvaluationRequest,
    http_request: Request,
    state: AppState = Depends(get_app_state),
) -> RagasEvaluationResponse:
    """通过接口执行 Ragas 评估。"""
    request_id = http_request.headers.get("x-request-id", str(uuid.uuid4()))
    cases = [
        MedicalRagasCase(
            question=item.question,
            tenant_id=item.tenant_id or request.tenant_id,
            grading_notes=item.grading_notes,
            reference=item.reference,
            metadata_filter=item.metadata_filter,
        )
        for item in request.cases
    ]
    try:
        results = evaluate_medical_rag_cases(
            pipeline=state.medical_pipeline,
            settings=settings,
            request_id=request_id,
            cases=cases,
            include_citations=request.include_citations,
        )
    except Exception as exc:
        logger.exception(
            "ragas_evaluation_failed",
            extra={"event": "ragas_evaluation_failed", "request_id": request_id, "case_count": len(cases)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    logger.info(
        "ragas_evaluation_completed",
        extra={
            "event": "ragas_evaluation_completed",
            "request_id": request_id,
            "case_count": len(results),
        },
    )
    return RagasEvaluationResponse(
        request_id=request_id,
        total_cases=len(results),
        results=[
            RagasEvaluationItem(
                question=item.question,
                answer=item.answer,
                citation_count=item.citation_count,
                duration_ms=item.duration_ms,
                metrics=item.metrics,
                metric_reasons=item.metric_reasons,
                metadata=item.metadata,
                citations=[
                    Citation(
                        chunk_id=citation["chunk_id"],
                        document_id=citation["document_id"],
                        source=citation.get("source"),
                        score=citation["score"],
                        text=citation["text"],
                        metadata=citation["metadata"],
                    )
                    for citation in item.citations
                ],
            )
            for item in results
        ],
    )
