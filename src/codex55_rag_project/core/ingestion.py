from __future__ import annotations

from dataclasses import dataclass

from codex55_rag_project.core.ports import Chunker, DocumentLoader, Embedder, VectorStore


@dataclass(frozen=True)
class IngestionStats:
    documents: int
    chunks: int


class IngestionPipeline:
    """文档索引编排。

    只负责串起 load -> chunk -> embed -> upsert，不关心文档来自哪里、向量写到哪里。
    """

    def __init__(self, loader: DocumentLoader, chunker: Chunker, embedder: Embedder, vector_store: VectorStore) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def run(self) -> IngestionStats:
        # 1. 读取原始文档；生产中 loader 可替换为对象存储、数据库、消息队列等。
        documents = self.loader.load()
        # 2. 切分成适合召回的 chunk；chunk 大小和 overlap 会直接影响召回质量。
        chunks = self.chunker.split(documents)
        # 3. 批量生成向量；这里保持批量调用，减少模型服务网络开销。
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        # 4. 写入向量库；chunk metadata 中的 tenant_id 会用于查询时隔离。
        self.vector_store.upsert(chunks, vectors)
        return IngestionStats(documents=len(documents), chunks=len(chunks))
