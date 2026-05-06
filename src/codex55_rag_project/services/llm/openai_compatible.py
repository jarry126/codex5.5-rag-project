"""OpenAI 兼容协议的 Embedding 和 Chat 实现。

中文：只依赖 OpenAI 协议形状，不依赖 OpenAI 官方服务；任何兼容 `/embeddings` 和 `/chat/completions` 的服务均可替换。
English: Depends on OpenAI wire protocol only; any compatible `/embeddings` and `/chat/completions` service can be swapped.
"""

from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from codex55_rag_project.core.ports import Vector
from codex55_rag_project.core.models import Answer, RetrievedChunk


class OpenAICompatibleEmbedder:
    """OpenAI-compatible embedding adapter.

    中文：这里只依赖 OpenAI 协议形状，不依赖 OpenAI 官方服务本身；DeepSeek、通义兼容模式、
    vLLM、自建网关只要实现 `/embeddings` 即可替换。
    English: This adapter depends on the OpenAI wire protocol, not the OpenAI hosted service.
    Any compatible provider can be swapped in by changing `base_url`, `api_key`, and model names.
    """

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 30.0) -> None:
        # 这里创建的是远程模型服务客户端，不是在本进程加载大模型。
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    @retry(wait=wait_exponential_jitter(initial=0.5, max=8.0), stop=stop_after_attempt(3))
    def embed_texts(self, texts: list[str]) -> list[Vector]:
        """批量文本向量化。

        中文：批量 embedding 可以显著降低网络开销；调用方负责按供应商 token 限制切批。
        English: Batch embeddings reduce network overhead; callers should batch by provider token limits.
        """
        # 中文：批量 embedding 可以显著降低网络开销；调用方负责按供应商 token 限制切批。
        # English: Batch embeddings reduce network overhead; callers should batch by provider token limits.
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class OpenAICompatibleChatLLM:
    """OpenAI-compatible chat completion adapter.

    中文：生成层只接收已经构造好的 prompt 和引用上下文，避免供应商 SDK 泄漏到核心 RAG 编排。
    English: Generation receives a built prompt plus retrieved contexts so provider SDK details stay out
    of the core RAG pipeline.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> None:
        # Chat LLM 同样只保存客户端和模型配置，真正调用发生在 generate。
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature

    @retry(wait=wait_exponential_jitter(initial=0.5, max=8.0), stop=stop_after_attempt(3))
    def generate(self, prompt: str, question: str, contexts: list[RetrievedChunk]) -> Answer:
        """基于 prompt 和上下文生成回答。

        中文：使用 tenacity 对网络抖动和服务短暂失败做最多 3 次重试。
        English: Uses tenacity for up to 3 retries on network jitter and brief service failures.
        """
        # tenacity 会对临时网络抖动/模型服务短暂失败做最多 3 次重试。
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "你是一个严谨的企业知识库问答助手。"},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        # contexts 原样返回给 API 层生成 citations，保证回答和引用来自同一批上下文。
        return Answer(
            question=question,
            text=text.strip(),
            contexts=contexts,
            metadata={
                "provider": "openai-compatible",
                "base_url": str(self.client.base_url),
                "model": self.model,
                "usage": response.usage.model_dump() if response.usage else None,
            },
        )
