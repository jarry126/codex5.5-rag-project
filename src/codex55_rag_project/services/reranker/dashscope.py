"""阿里云 DashScope Reranker 实现。

中文：使用 DashScope 专门的 rerank API 对检索结果重排序，提高相关性。
English: Uses DashScope's dedicated rerank API to reorder retrieval results for higher relevance.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from codex55_rag_project.core.models import RetrievedChunk


class DashScopeReranker:
    """Alibaba Cloud DashScope reranker.

    中文：阿里云重排序接口不属于 OpenAI chat/embedding 标准路径，所以单独封装；失败时由调用方决定是否降级。
    English: DashScope rerank is not part of the OpenAI chat/embedding path, so it is wrapped separately.
    """

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 30.0) -> None:
        # 使用 OpenAI SDK 的 post 方法调用 DashScope rerank endpoint。
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    # @retry：stop_after_attempt(3) 最多重试3次。wait_exponential_jitter(initial=0.5, max=8.0)：指数增加，随机抖动
    @retry(wait=wait_exponential_jitter(initial=0.5, max=8.0), stop=stop_after_attempt(3))
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """对候选切片重排序。

        中文：将候选切片文本发送给 DashScope rerank API，返回按相关性排序的切片列表。
        English: Sends candidate texts to DashScope rerank API, returns relevance-sorted chunks.
        """
        if not candidates:
            return []
        # DashScope rerank API 不是标准 OpenAI endpoint，使用 client.post 直接调用。
        response = self.client.post(
            "/reranks",
            cast_to=object,
            body={
                "model": self.model,
                "query": query,
                "documents": [candidate.chunk.text for candidate in candidates],
                "top_n": top_k,
                "return_documents": False,
            },
        )
        ranked = _extract_rerank_results(response)
        # 如果 rerank 返回空结果，降级返回原始候选切片的前 top_k 个。
        if not ranked:
            return candidates[:top_k]
        output: list[RetrievedChunk] = []
        # 根据 rerank 结果的 index 重组候选切片，并更新 score。
        for item in ranked[:top_k]:
            index = int(item["index"])
            score = float(item.get("relevance_score", candidates[index].score))
            candidate = candidates[index]
            output.append(RetrievedChunk(chunk=candidate.chunk, score=score))
        return output


def _extract_rerank_results(response: Any) -> list[dict[str, Any]]:
    """从 rerank 响应中提取结果列表。

    中文：处理不同格式的响应（dict、pydantic model、object），统一返回 results 列表。
    English: Handles different response formats (dict, pydantic model, object), returns results list.
    """
    if isinstance(response, dict):
        return list(response.get("results", []))
    if hasattr(response, "model_dump"):
        data = response.model_dump()
        return list(data.get("results", []))
    return list(getattr(response, "results", []))

