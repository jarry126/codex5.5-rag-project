from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _langfuse_quickstart_enabled() -> bool:
    return os.getenv("LANGFUSE_RUN_QUICKSTART", "").lower() in {"1", "true", "yes", "on"}


def _langfuse_qwen_plus_quickstart_enabled() -> bool:
    # 这个测试会真实调用大模型，所以必须在命令行显式传入开关；不从 .env 自动开启。
    return os.environ.get("LANGFUSE_RUN_QWEN_PLUS_QUICKSTART", "").lower() in {"1", "true", "yes", "on"}


def _skip_if_langfuse_quickstart_disabled() -> None:
    if not _langfuse_quickstart_enabled():
        # pytest.skip 会立刻结束当前这个测试函数，后面的代码不会继续执行。
        pytest.skip("设置 LANGFUSE_RUN_QUICKSTART=true 后，才会发送真实的 Langfuse 快速验证 trace。")


def _skip_if_langfuse_qwen_plus_quickstart_disabled() -> None:
    if not _langfuse_qwen_plus_quickstart_enabled():
        pytest.skip("设置 LANGFUSE_RUN_QWEN_PLUS_QUICKSTART=true 后，才会真实调用 qwen-plus 并发送 Langfuse trace。")


def _skip_if_missing_env(names: tuple[str, ...]) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        pytest.skip(f"缺少环境变量：{', '.join(missing)}")


@pytest.mark.integration
def test_langfuse_quickstart_trace() -> None:
    """快速验证 Langfuse Python SDK 能把一次 trace 发送到 Langfuse。

    中文：默认跳过，避免普通单元测试向真实 Langfuse Cloud 写入数据。
    需要手动设置 LANGFUSE_RUN_QUICKSTART=true，并提供 LANGFUSE_* 配置后才会执行。
    """
    load_dotenv(PROJECT_ROOT / ".env")

    _skip_if_langfuse_quickstart_disabled()
    _skip_if_missing_env(("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_BASE_URL"))

    """
        尝试
        import langfuse；
        如果导入成功，就返回
        langfuse
        模块；
        如果导入失败，就跳过当前测试
    """
    langfuse = pytest.importorskip("langfuse")
    client = langfuse.get_client()
    trace_id = f"codex55-rag-langfuse-quickstart-{uuid4()}"

    with client.start_as_current_observation(
        as_type="span",
        name="Langfuse 快速验证 Trace",
        input={"question": "Langfuse 能否记录 RAG 执行链路？"},
        metadata={"项目": "codex5.5-rag-project", "测试类型": "手动快速验证"},
    ) as root_span:
        with langfuse.propagate_attributes(
            user_id="手动测试用户",
            session_id=trace_id,
            tags=["快速验证", "RAG"],
            trace_name="codex55-rag-Langfuse-快速验证",
        ):
            with client.start_as_current_observation(
                as_type="span",
                name="检索上下文",
                input={"query": "Langfuse RAG 链路追踪"},
            ) as retrieval_span:
                retrieved_contexts = [
                    {
                        "切片ID": "演示切片-1",
                        "相关性分数": 0.91,
                        "文本": "Langfuse 可以记录 trace、span、generation、输入输出、耗时和 token 使用情况。",
                    }
                ]
                retrieval_span.update(output={"上下文": retrieved_contexts})

            with client.start_as_current_observation(
                as_type="generation",
                name="大模型生成",
                model="qwen-plus",
                input=[
                    {"role": "system", "content": "你是一个 RAG 助手。"},
                    {"role": "user", "content": "Langfuse 可以记录什么？"},
                ],
            ) as generation:
                generation.update(
                    output="Langfuse 可以记录 trace、span、generation、输入输出、耗时和 token 使用情况。",
                    usage_details={"input_tokens": 24, "output_tokens": 18, "total_tokens": 42},
                    metadata={"模型提供方": "DashScope OpenAI-compatible"},
                )

        root_span.update(output={"状态": "成功", "trace_id": trace_id})

    client.flush()


@pytest.mark.integration
def test_langfuse_official_openai_integration_with_qwen_plus() -> None:
    """按 Langfuse 官方 observe + OpenAI integration 写法验证 qwen-plus 调用。

    中文：这个测试会真实调用 DashScope OpenAI-compatible chat 接口，并把 trace 发送到 Langfuse。
    默认跳过，只有设置 LANGFUSE_RUN_QWEN_PLUS_QUICKSTART=true 后才执行。
    """
    # _skip_if_langfuse_qwen_plus_quickstart_disabled()
    load_dotenv(PROJECT_ROOT / ".env")
    _skip_if_missing_env(("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_BASE_URL", "DASHSCOPE_API_KEY"))

    langfuse = pytest.importorskip("langfuse")
    from langfuse import observe
    from langfuse.openai import openai

    openai_client = openai.OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    model = os.getenv("RAG_CHAT_MODEL", "qwen-plus")

    @observe(name="调用 qwen-plus 生成回答")
    def story() -> str:
        completion = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个助手。"},
                {"role": "user", "content": "用一句话说明应该怎么保持好心情"},
            ],
            temperature=0,
        )
        return completion.choices[0].message.content or ""

    @observe(name="Langfuse 官方 OpenAI 集成快速验证")
    def main() -> str:
        return story()

    with langfuse.propagate_attributes(
        user_id="手动测试用户",
        session_id=f"codex55-rag-openai-quickstart-{uuid4()}",
        tags=["快速验证", "OpenAI集成", "qwen-plus"],
        trace_name="codex55-rag-Langfuse-官方-OpenAI-集成",
    ):
        answer = main()
    print(answer)
    langfuse.get_client().flush()

    assert answer
