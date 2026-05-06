"""引用式 Prompt 构建器。

中文：构建带有上下文编号的 Prompt，引导 LLM 在回答中引用来源。
English: Builds prompts with numbered contexts, guiding LLM to cite sources in responses.
"""

from __future__ import annotations

from codex55_rag_project.core.models import RetrievedChunk


class CitationPromptBuilder:
    """引用式 Prompt 构建器。

    中文：集中管理回答风格、引用格式和"不知道"策略，避免散落在 API 层。
    English: Centralizes answer style, citation format, and abstention policy outside the API layer.
    """

    def build(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """构建带引用的 Prompt。

        中文：将上下文切片编号，要求 LLM 只根据给定上下文回答，并引用编号。
        English: Numbers context chunks, requires LLM to answer only from given contexts and cite numbers.
        """
        # 将上下文切片编号，便于 LLM 在回答中引用 [1], [2] 等。
        context_text = "\n\n".join(
            f"[{index}] source={item.chunk.metadata.get('source', item.chunk.document_id)}\n{item.chunk.text}"
            for index, item in enumerate(contexts, start=1)
        )
        # Prompt 定义角色、约束和格式，确保回答严谨且有引用。
        return (
            "你是一个严谨的企业知识库问答助手。只根据给定上下文回答；"
            "如果上下文不足，明确说明无法从资料中确认。\n\n"
            f"问题：{question}\n\n"
            f"上下文：\n{context_text}\n\n"
            "回答时给出简洁结论，并引用相关上下文编号。"
        )

