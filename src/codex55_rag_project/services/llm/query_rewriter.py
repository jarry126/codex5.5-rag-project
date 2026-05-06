"""Question rewriting for multi-query retrieval."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import SecretStr


class QueryRewriter(Protocol):
    def rewrite(self, question: str, count: int) -> list[str]:
        """Return rewritten retrieval queries."""


class DashScopeQueryRewriter:
    """Uses a chat model to produce retrieval-oriented question rewrites."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> None:
        from langchain_openai import ChatOpenAI

        self.client = ChatOpenAI(
            api_key=SecretStr(api_key),
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )

    def rewrite(self, question: str, count: int) -> list[str]:
        prompt = (
            "请将用户问题改写为适合医学知识库检索的不同表达。"
            "只返回 JSON 数组，不要解释。"
            f"需要 {count} 条，保留原意，避免编造新症状。\n\n"
            f"用户问题：{question}"
        )
        response = self.client.invoke(
            [
                ("system", "你是一个严谨的医学 RAG 检索问题改写器。"),
                ("user", prompt),
            ]
        )
        return _parse_rewrites(str(response.content), count)


class NoopQueryRewriter:
    """Fallback rewriter used by tests and local-only paths."""

    def rewrite(self, question: str, count: int) -> list[str]:
        return []


def rewrite_with_fallback(rewriter: QueryRewriter, question: str, count: int) -> list[str]:
    try:
        rewrites = rewriter.rewrite(question, count)
    except Exception:
        return [question]
    queries = [question]
    for rewrite in rewrites:
        normalized = rewrite.strip()
        if normalized and normalized not in queries:
            queries.append(normalized)
    return queries[: count + 1]


def _parse_rewrites(content: str, count: int) -> list[str]:
    text = content.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()][:count]
    except json.JSONDecodeError:
        pass
    lines = [line.strip(" -\t\r\n") for line in text.splitlines()]
    return [line for line in lines if line][:count]
