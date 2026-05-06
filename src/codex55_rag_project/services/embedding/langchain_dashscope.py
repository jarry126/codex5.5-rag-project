"""DashScope embedding adapter.

中文：通过 OpenAI-compatible endpoint 调用 DashScope embedding，并实现 LangChain Embeddings 协议。
English: Calls DashScope embeddings through the OpenAI-compatible endpoint and implements the LangChain protocol.
"""

from __future__ import annotations

from openai import OpenAI

from codex55_rag_project.config import Settings


class DashScopeEmbeddings:
    """LangChain-compatible DashScope embedding adapter.

    DashScope text-embedding-v4 supports the OpenAI-compatible endpoint, but production usage needs two provider
    details to be explicit: batch size must not exceed 10 and encoding_format should be float.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        timeout: float,
        batch_size: int = 10,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            vectors.extend([list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)])
        return vectors

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def build_dashscope_embeddings(settings: Settings) -> object:
    """构建 LangChain-compatible DashScope Embeddings 实例。"""

    return DashScopeEmbeddings(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout=settings.request_timeout_seconds,
    )
