from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codex55_rag_project.core.ports import LLM, PromptBuilder, Reranker, Retriever
from codex55_rag_project.core.models import Answer


@dataclass(frozen=True)
class RetrievalConfig:
    # 先粗召回多少个，默认12个
    candidate_k: int = 12
    # 最终喂给大模型的数据，默认4个
    final_k: int = 4
    # 低于多少分的不要，默认0
    min_score: float = 0.0


class RagPipeline:
    """RAG query orchestrator.

    中文：Pipeline 只编排检索、重排、Prompt 和生成，不直接依赖数据库或模型 SDK。
    English: The pipeline orchestrates retrieval, reranking, prompting, and generation without importing
    database or model SDKs.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        prompt_builder: PromptBuilder,
        llm: LLM,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.config = config or RetrievalConfig()

    def ask(
        self,
        question: str,
        tenant_id: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> Answer:
        # 中文：tenant_id 和 metadata_filter 必须从查询入口一路传到向量库，防止跨租户数据泄漏。
        # English: tenant_id and metadata_filter must flow to the vector store to prevent cross-tenant leaks.
        # 第一步先召回较大的候选集 candidate_k，给后面的 reranker 留出筛选空间。
        candidates = self.retriever.retrieve(
            question,
            top_k=self.config.candidate_k,
            tenant_id=tenant_id,
            metadata_filter=metadata_filter,
        )
        # 中文：min_score 用于截断明显不相关上下文；阈值需要通过离线评测校准。
        # English: min_score removes weak contexts; calibrate it with offline retrieval evaluation.
        candidates = [item for item in candidates if item.score >= self.config.min_score]
        # reranker 将候选集压缩到 final_k，减少 prompt 噪声和 token 消耗。
        contexts = self.reranker.rerank(question, candidates, top_k=self.config.final_k)
        # PromptBuilder 统一管理引用格式、回答边界和“不知道”策略，避免散落在 API 层。
        prompt = self.prompt_builder.build(question, contexts)
        # LLM 只负责基于 prompt 生成回答，不参与检索、权限过滤或上下文选择。
        return self.llm.generate(prompt=prompt, question=question, contexts=contexts)
