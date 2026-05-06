"""LangChain ChatOpenAI 封装，指向阿里云 DashScope OpenAI-compatible endpoint。

中文：使用 LangChain 的 ChatOpenAI 客户端，通过 base_url 指向阿里云，保持与 OpenAI SDK 兼容。
English: Uses LangChain's ChatOpenAI client pointed at DashScope via base_url, maintaining OpenAI SDK compatibility.
"""

from __future__ import annotations

from pydantic import SecretStr

from codex55_rag_project.core.models import Answer, RetrievedChunk


class LangChainDashScopeChatLLM:
    """LangChain ChatOpenAI wrapper for Alibaba Cloud's OpenAI-compatible endpoint.

    中文：封装 LangChain ChatOpenAI，通过 DashScope OpenAI-compatible API 生成回答。
    English: Wraps LangChain ChatOpenAI to generate answers via DashScope's OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> None:
        from langchain_openai import ChatOpenAI

        self.model = model
        # 使用 LangChain ChatOpenAI 客户端，通过 base_url 指向阿里云 DashScope。
        self.client = ChatOpenAI(
            api_key=SecretStr(api_key),
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )

    def generate(self, prompt: str, question: str, contexts: list[RetrievedChunk]) -> Answer:
        """基于 prompt 和上下文生成回答。

        中文：system message 定义角色，user message 是构造好的 prompt；返回 Answer 对象。
        English: System message defines role; user message is the constructed prompt; returns Answer object.
        """
        response = self.client.invoke(
            [
                ("system", "你是一个严谨的企业知识库问答助手。"),
                ("user", prompt),
            ]
        )
        # 将 LangChain 响应转换为内部 Answer 模型，保留 provider/framework/model 元数据。
        return Answer(
            question=question,
            text=str(response.content).strip(),
            contexts=contexts,
            metadata={
                "provider": "dashscope-openai-compatible",
                "framework": "langchain",
                "model": self.model,
            },
        )
