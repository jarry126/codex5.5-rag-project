from __future__ import annotations

from typing import Any, Protocol

from codex55_rag_project.core.models import Answer, Chunk, Document, RetrievedChunk


Vector = list[float]

"""
    任何对象，只要有 split(documents) 方法，并且返回 list[Chunk]，
    它就可以被当成 Chunker 使用。

"""

class DocumentLoader(Protocol):
    """Source document reader.

    中文：生产中可替换为对象存储、数据库、网页爬虫、CMS 或消息队列 loader。
    English: Replace this with object storage, database, crawler, CMS, or queue-backed loaders.
    """

    def load(self) -> list[Document]:
        """Load source documents."""


class Chunker(Protocol):
    """Document-to-chunk splitter.

    中文：切分策略会直接影响召回质量；建议按文档类型提供不同实现。
    English: Chunking quality strongly affects retrieval; provide document-type-specific implementations.
    """

    def split(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into retrieval units."""


class Embedder(Protocol):
    """Text embedding provider boundary.

    中文：业务代码只依赖向量结果，不依赖 OpenAI、bge、vLLM 等具体 SDK。
    English: Business code depends on vectors, not a concrete OpenAI, bge, vLLM, or vendor SDK.
    """

    def embed_texts(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of texts into vectors."""


class VectorStore(Protocol):
    """Vector persistence and search boundary.

    中文：可替换为 pgvector、Milvus、Qdrant、Elasticsearch 等；接口必须保留 metadata 过滤能力。
    English: Swap in pgvector, Milvus, Qdrant, Elasticsearch, etc.; keep metadata filtering available.
    """

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        """Persist chunks and their vectors."""

    def search(
        self,
        query_vector: Vector,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the closest chunks."""


class Retriever(Protocol):
    """Candidate retrieval boundary.

    中文：这里可以扩展 hybrid search、query rewrite、多路召回和权限过滤。
    English: Extend here for hybrid search, query rewriting, multi-retrieval, and authorization filters.
    """

    def retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve candidate chunks for a query."""


class Reranker(Protocol):
    """Candidate reranking boundary.

    中文：第一版可用轻量 reranker，生产高精度场景可替换 cross-encoder 或 LLM reranker。
    English: Start lightweight; replace with a cross-encoder or LLM reranker for higher precision.
    """

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Rerank retrieved candidates."""


class PromptBuilder(Protocol):
    """Prompt construction boundary.

    中文：集中管理回答风格、引用格式和“不知道”策略，避免散落在 API 层。
    English: Centralizes answer style, citation format, and abstention policy outside the API layer.
    """

    def build(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """Build the final LLM prompt."""


class LLM(Protocol):
    """Generation provider boundary.

    中文：LLM 实现只负责生成，不负责检索、权限或上下文选择。
    English: LLM implementations generate only; retrieval, authorization, and context selection stay upstream.
    """

    def generate(self, prompt: str, question: str, contexts: list[RetrievedChunk]) -> Answer:
        """Generate an answer from prompt and contexts."""
