"""RAG 核心领域模型。

中文：这些 dataclass 是整个 RAG 系统的基本数据单元，贯穿索引、检索、生成三个阶段。
English: These dataclasses are the fundamental data units across ingestion, retrieval, and generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Metadata = dict[str, Any]


@dataclass(frozen=True)
class Document:
    """原始文档。

    中文：来自外部系统的源文档，可附带来源、作者、租户等元数据。
    English: Source document from external systems, with optional source/author/tenant metadata.
    """

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """文档切片。

    中文：一个 Document 切成多个 Chunk，每个 Chunk 是检索和召回的基本单位。
    English: A Document is split into multiple Chunks; each Chunk is the retrieval unit.
    """

    id: str
    document_id: str
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """检索到的带分数切片。

    中文：向量检索返回的结果，score 表示与查询的相似度，越大越相关。
    English: Retrieved result with similarity score; higher score means more relevant.
    """

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    """RAG 生成的回答。

    中文：包含问题原文、生成答案、引用的上下文切片和生成元数据。
    English: Contains the original question, generated answer, cited contexts, and metadata.
    """

    question: str
    text: str
    contexts: list[RetrievedChunk]
    metadata: Metadata = field(default_factory=dict)

