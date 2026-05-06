"""本地测试 RAG 服务构建器。

中文：组装无外部依赖的 RAG 链路，使用内存向量存储和本地向量化器，适合单元测试和演示。
English: Assembles RAG chain without external deps, using in-memory vector store and local embedder for tests and demos.
"""

from __future__ import annotations

from codex55_rag_project.services.chunking import RecursiveTextChunker
from codex55_rag_project.services.embedding.local_hash import HashingEmbedder
from codex55_rag_project.core.ingestion import IngestionPipeline
from codex55_rag_project.services.llm.local_extract import ExtractiveLLM
from codex55_rag_project.loaders.text import StaticDocumentLoader
from codex55_rag_project.core.pipeline import RagPipeline, RetrievalConfig
from codex55_rag_project.services.prompt.citation import CitationPromptBuilder
from codex55_rag_project.services.reranker.keyword import KeywordOverlapReranker
from codex55_rag_project.services.retriever.vector_retriever import VectorRetriever
from codex55_rag_project.core.models import Document
from codex55_rag_project.data.vector_store.in_memory import InMemoryVectorStore


def build_local_rag(documents: list[Document]) -> RagPipeline:
    """构建本地无外部依赖的 RAG 链路。

    中文：使用 HashingEmbedder 和 InMemoryVectorStore，适合单元测试和演示。
    English: Uses HashingEmbedder and InMemoryVectorStore, suitable for unit tests and demos.
    """
    # 使用哈希向量化器，不调用外部模型
    embedder = HashingEmbedder(dimensions=256)
    # 使用内存向量存储，不依赖数据库
    vector_store = InMemoryVectorStore()
    # 使用静态文档加载器，直接传入文档列表
    loader = StaticDocumentLoader(documents)
    # 文本切分器，chunk_size 和 overlap 较小适合测试
    chunker = RecursiveTextChunker(chunk_size=500, overlap=80)
    # 执行索引流程
    IngestionPipeline(loader, chunker, embedder, vector_store).run()

    # 构建检索器
    retriever = VectorRetriever(embedder=embedder, vector_store=vector_store)
    # 构建 RAG Pipeline，使用关键词重排序和提取式生成器
    return RagPipeline(
        retriever=retriever,
        reranker=KeywordOverlapReranker(),
        prompt_builder=CitationPromptBuilder(),
        llm=ExtractiveLLM(),
        config=RetrievalConfig(candidate_k=8, final_k=3),
    )

