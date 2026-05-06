from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codex55_rag_project.bootstrap.local_factory import build_local_rag
from codex55_rag_project.core.models import Document


def main() -> None:
    documents = [
        Document(
            id="architecture",
            text=(
                "RAG 系统通常包含数据接入、文本切分、向量化、索引、检索、重排、提示词构造和生成。"
                "为了可扩展，应通过接口隔离 Embedding、VectorStore、Retriever、Reranker 和 LLM。"
            ),
            metadata={"source": "architecture-note"},
        ),
        Document(
            id="production",
            text=(
                "生产级 RAG 需要支持租户隔离、权限过滤、增量索引、可观测性、召回评测和用户反馈闭环。"
                "向量库可以选择 Milvus、Qdrant、Weaviate、pgvector 或 Elasticsearch。"
            ),
            metadata={"source": "production-note"},
        ),
    ]
    rag = build_local_rag(documents)
    answer = rag.ask("可扩展 RAG 架构应该隔离哪些组件？")
    print(answer.text)


if __name__ == "__main__":
    main()
