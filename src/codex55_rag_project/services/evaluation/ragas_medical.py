"""Ragas 医疗 RAG 评估。

这个模块只在需要评估时才懒加载 ragas，避免普通 API 启动被评估依赖影响。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from codex55_rag_project.config.settings import Settings
from codex55_rag_project.core.medical_pipeline import MedicalRagPipeline


@dataclass(frozen=True)
class MedicalRagasCase:
    """单条医疗 RAG 评估用例。"""

    question: str
    tenant_id: str
    grading_notes: str = ""
    reference: str = ""
    metadata_filter: dict[str, Any] | None = None


@dataclass(frozen=True)
class MedicalRagasResult:
    """单条医疗 RAG 评估结果。"""

    question: str
    answer: str
    citation_count: int
    duration_ms: float
    metrics: dict[str, float]
    metric_reasons: dict[str, str]
    metadata: dict[str, Any]
    citations: list[dict[str, Any]]


def evaluate_medical_rag_cases(
    *,
    pipeline: MedicalRagPipeline,
    settings: Settings,
    request_id: str,
    cases: list[MedicalRagasCase],
    include_citations: bool = False,
) -> list[MedicalRagasResult]:
    """执行医疗 RAG 用例并用 Ragas 打分。

    中文：每条 case 会先走真实 medical_pipeline，即真实查询 Postgres/pgvector；
    Ragas 只在拿到 answer 和 citations 后进行质量评估。
    """
    judge_llm = _build_judge_llm(settings)
    results: list[MedicalRagasResult] = []

    for index, case in enumerate(cases, start=1):
        item_request_id = f"{request_id}-{index}"
        started = time.perf_counter()
        answer = pipeline.ask(
            question=case.question,
            request_id=item_request_id,
            tenant_id=case.tenant_id,
            metadata_filter=case.metadata_filter,
        )
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        citations = [
            {
                "chunk_id": item.chunk.id,
                "document_id": item.chunk.document_id,
                "source": item.chunk.metadata.get("source"),
                "score": item.score,
                "text": item.chunk.text,
                "metadata": item.chunk.metadata,
            }
            for item in answer.contexts
        ]
        contexts_text = "\n\n".join(
            f"[上下文 {context_index + 1}]\n{item['text']}" for context_index, item in enumerate(citations)
        )
        metric_values: dict[str, float] = {}
        metric_reasons: dict[str, str] = {}
        metrics = _build_metrics(judge_llm, include_reference_metrics=bool(case.reference))
        for metric in metrics:
            score = asyncio.run(
                _score_metric(
                    metric,
                    judge_llm,
                    {
                        "user_input": case.question,
                        "response": answer.text,
                        "retrieved_contexts": contexts_text,
                        "grading_notes": case.grading_notes,
                        "reference": case.reference,
                    },
                )
            )
            metric_values[metric.name] = _score_value(score)
            metric_reasons[metric.name] = _score_reason(score)

        results.append(
            MedicalRagasResult(
                question=case.question,
                answer=answer.text,
                citation_count=len(citations),
                duration_ms=duration_ms,
                metrics=metric_values,
                metric_reasons=metric_reasons,
                metadata=answer.metadata,
                citations=citations if include_citations else [],
            )
        )
    return results


def _build_judge_llm(settings: Settings) -> Any:
    """构建 Ragas 评分用 LLM。"""
    try:
        from openai import OpenAI
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError('缺少 Ragas 评估依赖，请先安装：pip install -e ".[dev]"') from exc

    if not settings.openai_api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY/RAG_OPENAI_API_KEY，无法调用 Ragas 评分模型")

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    return llm_factory(settings.chat_model, provider="openai", client=client)


def _new_numeric_metric(numeric_metric_class: Any, judge_llm: Any, **kwargs: Any) -> Any:
    """兼容不同 Ragas 小版本的 NumericMetric 构造参数。"""
    try:
        return numeric_metric_class(**kwargs, llm=judge_llm)
    except TypeError:
        return numeric_metric_class(**kwargs)


def _build_metrics(judge_llm: Any, include_reference_metrics: bool) -> list[Any]:
    """定义接口评估使用的 0~1 Ragas 数字指标。"""
    try:
        from ragas.metrics import NumericMetric
    except ImportError as exc:
        raise RuntimeError('缺少 Ragas 评估依赖，请先安装：pip install -e ".[dev]"') from exc

    faithfulness = _new_numeric_metric(
        NumericMetric,
        judge_llm,
        name="faithfulness",
        allowed_values=(0.0, 1.0),
        prompt=(
            "你是 RAG 评估员。请给回答的忠实度打 0 到 1 分。\n"
            "评分含义：1 表示回答的关键结论完全被检索上下文支持；"
            "0 表示回答明显脱离上下文或编造；中间值表示部分支持。\n"
            "问题：{user_input}\n"
            "回答：{response}\n"
            "检索上下文：{retrieved_contexts}\n"
            "只返回 0 到 1 之间的数值和简短理由。"
        ),
    )
    answer_relevancy = _new_numeric_metric(
        NumericMetric,
        judge_llm,
        name="answer_relevancy",
        allowed_values=(0.0, 1.0),
        prompt=(
            "你是 RAG 评估员。请给回答相关性打 0 到 1 分。\n"
            "评分含义：1 表示回答完整、直接地回答了用户问题；"
            "0 表示回答和问题无关；中间值表示部分相关。\n"
            "问题：{user_input}\n"
            "回答：{response}\n"
            "评分说明：{grading_notes}\n"
            "只返回 0 到 1 之间的数值和简短理由。"
        ),
    )
    context_quality = _new_numeric_metric(
        NumericMetric,
        judge_llm,
        name="context_quality",
        allowed_values=(0.0, 1.0),
        prompt=(
            "你是 RAG 检索评估员。请给检索上下文质量打 0 到 1 分。\n"
            "评分含义：1 表示检索上下文足以支撑回答该问题；"
            "0 表示上下文和问题无关或没有有效信息；中间值表示部分有用。\n"
            "问题：{user_input}\n"
            "检索上下文：{retrieved_contexts}\n"
            "标准答案或评分说明：{reference}\n{grading_notes}\n"
            "只返回 0 到 1 之间的数值和简短理由。"
        ),
    )
    metrics = [faithfulness, answer_relevancy, context_quality]
    if include_reference_metrics:
        metrics.append(
            _new_numeric_metric(
                NumericMetric,
                judge_llm,
                name="reference_correctness",
                allowed_values=(0.0, 1.0),
                prompt=(
                    "你是 RAG 答案评估员。请比较回答和标准答案，给事实正确性打 0 到 1 分。\n"
                    "评分含义：1 表示回答和标准答案事实一致且覆盖主要要点；"
                    "0 表示回答与标准答案冲突或完全没有覆盖；中间值表示部分正确。\n"
                    "问题：{user_input}\n"
                    "回答：{response}\n"
                    "标准答案：{reference}\n"
                    "只返回 0 到 1 之间的数值和简短理由。"
                ),
            )
        )
    return metrics


async def _score_metric(metric: Any, judge_llm: Any, sample: dict[str, Any]) -> Any:
    """兼容 Ragas metric 的同步/异步评分方法。

    中文：这里优先使用同步 score，因为 _build_judge_llm 创建的是同步 OpenAI client；
    如果用 ascore，Ragas 会调用 agenerate，从而要求 AsyncOpenAI client。
    """
    if hasattr(metric, "score"):
        try:
            return metric.score(**sample, llm=judge_llm)
        except TypeError:
            return metric.score(**sample)
    if hasattr(metric, "ascore"):
        try:
            return await metric.ascore(**sample, llm=judge_llm)
        except TypeError:
            return await metric.ascore(**sample)
    raise RuntimeError(f"Ragas 指标 {metric!r} 没有 score/ascore 方法")


def _score_value(score: Any) -> float:
    """提取 Ragas 分值。"""
    if hasattr(score, "value"):
        return _clamp_score(float(score.value))
    return _clamp_score(float(score))


def _score_reason(score: Any) -> str:
    """提取 Ragas 评分理由。"""
    return str(getattr(score, "reason", "") or "")


def _clamp_score(value: float) -> float:
    """把评分限制在 0~1，避免模型偶尔返回越界值。"""
    return round(min(1.0, max(0.0, value)), 4)
