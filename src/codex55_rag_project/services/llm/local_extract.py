"""本地降级生成器（用于测试和演示）。

中文：不调用外部模型，直接从上下文提取片段返回，保证演示可运行且结果可预测。
English: No external model calls, extracts from contexts directly, ensuring demos run with predictable results.
"""

from __future__ import annotations

from codex55_rag_project.core.models import Answer, RetrievedChunk


class ExtractiveLLM:
    """Local fallback generator.

    中文：不调用外部模型，直接从上下文提取片段返回，保证演示可运行且结果可预测。
    English: No external model calls, extracts from contexts directly, ensuring demos run with predictable results.
    Production code should provide an adapter that calls a real chat/completion model.
    """

    def generate(self, prompt: str, question: str, contexts: list[RetrievedChunk]) -> Answer:
        """基于上下文生成回答（提取式）。

        中文：如果无上下文则返回"无法确认"；否则将每个切片截断后编号返回。
        English: Returns "无法确认" if no contexts; otherwise truncates and numbers each chunk.
        """
        if not contexts:
            return Answer(question=question, text="无法从资料中确认。", contexts=[])

        # 将每个上下文切片截断到 220 字符，避免过长。
        lines = []
        for index, item in enumerate(contexts, start=1):
            snippet = item.chunk.text.replace("\n", " ").strip()
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            lines.append(f"[{index}] {snippet}")

        return Answer(
            question=question,
            text="根据检索到的资料：\n" + "\n".join(lines),
            contexts=contexts,
            metadata={"prompt": prompt},
        )

