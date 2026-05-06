"""LangChain 集成的文档索引流程。

中文：使用 LangChain PGVector 的 add_documents 方法，在入库时自动调用 embedding。
English: Uses LangChain PGVector's add_documents method, embedding is called automatically during ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from codex55_rag_project.core.ports import Chunker, DocumentLoader
from codex55_rag_project.core.models import Chunk


class ChunkStore(Protocol):
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Persist chunks."""


@dataclass(frozen=True)
class IngestionStats:
    """索引统计结果。

    中文：记录入库的文档数和切分后的 chunk 数。
    English: Records document and chunk counts after ingestion.
    """

    documents: int
    chunks: int


class LangChainIngestionPipeline:
    """LangChain-backed document ingestion.

    中文：生产索引不再先手工生成 vectors，而是交给 LangChain PGVector 在 add_documents 时调用 embedding。
    English: Production ingestion does not precompute vectors manually; LangChain PGVector embeds documents
    during add_documents.
    """

    def __init__(self, loader: DocumentLoader, chunker: Chunker, vector_store: ChunkStore) -> None:
        # loader 读取文档，chunker 切分，vector_store 自动向量化并入库。
        self.loader = loader
        self.chunker = chunker
        self.vector_store = vector_store

    def run(self) -> IngestionStats:
        """执行索引流程。

        中文：load -> split -> add_chunks，返回索引统计。
        English: load -> split -> add_chunks, returns ingestion stats.
        """
        # 1. 加载源文档
        documents = self.loader.load()
        # 2. 切分成 chunk
        chunks = self.chunker.split(documents)
        # 3. LangChain PGVector 在 add_chunks 时自动调用 embedding 并入库
        self.vector_store.add_chunks(chunks)
        return IngestionStats(documents=len(documents), chunks=len(chunks))
