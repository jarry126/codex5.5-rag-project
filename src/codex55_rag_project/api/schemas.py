"""FastAPI HTTP 请求/响应模型。

中文：定义 ingest 和 query API 的输入输出结构，与 core/models 分层，便于 API 层独立演进。
English: Defines ingest/query API input/output schemas, separated from core/models for independent API evolution.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """用户提交的源文档。

    中文：id 和 text 是必填字段，metadata 可携带来源、作者等信息。
    English: id and text are required; metadata carries optional source/author info.
    """

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """文档入库请求。

    中文：tenant_id 用于多租户隔离，documents 是要入库的文档列表。
    English: tenant_id isolates data per tenant; documents is the list to ingest.
    """

    tenant_id: str = Field(default="default", min_length=1)
    documents: list[SourceDocument] = Field(min_length=1)


class IngestResponse(BaseModel):
    """文档入库响应。

    中文：返回入库的文档数和切分后的 chunk 数，便于用户确认。
    English: Returns document and chunk counts for user confirmation.
    """

    documents: int
    chunks: int


class QueryRequest(BaseModel):
    """问答请求。

    中文：question 是用户问题，tenant_id 和 metadata_filter 用于检索过滤。
    English: question is the user query; tenant_id and metadata_filter restrict retrieval.
    """

    tenant_id: str = Field(default="default", min_length=1)
    question: str = Field(min_length=1)
    metadata_filter: dict[str, Any] | None = None


class Citation(BaseModel):
    """回答引用的上下文切片。

    中文：包含 chunk_id、document_id、来源、相似度分数、文本内容和元数据。
    English: Contains chunk_id, document_id, source, score, text, and metadata for context citation.
    """

    chunk_id: str
    document_id: str
    source: str | None = None
    score: float
    text: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    """问答响应。

    中文：answer 是生成的回答，citations 是引用的上下文切片列表，metadata 包含生成元信息。
    English: answer is the generated response; citations list referenced contexts; metadata holds generation info.
    """

    answer: str
    citations: list[Citation]
    metadata: dict[str, Any]


class PdfIngestionRequest(BaseModel):
    """PDF 入库请求。"""

    tenant_id: str = Field(default="default", min_length=1)
    pdf_path: str | None = None


class PdfIngestionResponse(BaseModel):
    """PDF 入库响应。"""

    ingestion_id: str
    documents: int
    pages: int
    chunks: int
    source: str


class MedicalQueryRequest(BaseModel):
    """医疗问答请求。"""

    tenant_id: str = Field(default="default", min_length=1)
    question: str = Field(min_length=1)
    metadata_filter: dict[str, Any] | None = None
