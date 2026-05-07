#!/usr/bin/env python3
"""使用 Ragas 调用真实医疗 RAG 接口做质量评估。

这个脚本不会直接绕过业务代码查询数据库，而是调用已启动服务的
`POST /v1/medical/query`。接口内部会执行问题改写、多路混合召回、
pgvector/pg_jieba 检索、重排序和 LLM 生成；脚本只负责把接口返回的
answer/citations 转成 Ragas 可以评估的数据。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


DEFAULT_CASES = [
    {
        "question": "失眠的中医辨证思路是什么？",
        "grading_notes": "回答应基于知识库引用内容说明辨证思路，不应脱离资料直接给出绝对诊断。",
    },
    {
        "question": "咳嗽在中医临床中通常需要关注哪些辨证要点？",
        "grading_notes": "回答应基于检索上下文说明辨证要点，并避免无依据地给出具体处方承诺。",
    },
]


def _load_environment() -> None:
    """加载本地环境变量。

    生产环境通常由 Docker/K8s 注入变量；本地执行脚本时会自动读取项目根目录
    下的 `.env` 和 `.env.production`，但这些文件不会提交到代码仓库。
    """
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.production", override=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 评估用例。每行至少包含 question，可选 grading_notes。"""
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not item.get("question"):
                raise ValueError(f"{path}:{line_number} 缺少 question 字段")
            cases.append(item)
    if not cases:
        raise ValueError(f"{path} 没有可用评估用例")
    return cases


def _load_cases(cases_file: str | None) -> list[dict[str, Any]]:
    """加载评估用例；未配置文件时使用脚本内置的少量冒烟用例。"""
    if not cases_file:
        return DEFAULT_CASES
    path = Path(cases_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _read_jsonl(path)


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    """发送 JSON POST 请求，并把非 2xx 响应转换成明确异常。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url=url,
        data=body,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RAG 接口返回 HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 RAG 接口: {exc.reason}") from exc


def _query_medical_api(case: dict[str, Any], api_base_url: str, api_key: str, tenant_id: str, timeout: float) -> dict[str, Any]:
    """调用真实 `/v1/medical/query` 接口，让服务内部完成 Postgres 检索。"""
    request_id = f"ragas-{uuid4()}"
    started = time.perf_counter()
    response = _post_json(
        url=f"{api_base_url.rstrip('/')}/v1/medical/query",
        payload={
            "tenant_id": case.get("tenant_id", tenant_id),
            "question": case["question"],
            "metadata_filter": case.get("metadata_filter"),
        },
        headers={"x-api-key": api_key, "x-request-id": request_id},
        timeout=timeout,
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    citations = response.get("citations", [])
    return {
        "user_input": case["question"],
        "response": response.get("answer", ""),
        "retrieved_contexts": [item.get("text", "") for item in citations if item.get("text")],
        "grading_notes": case.get("grading_notes") or case.get("reference") or case.get("expected_answer") or "",
        "metadata": {
            "request_id": request_id,
            "duration_ms": duration_ms,
            "tenant_id": case.get("tenant_id", tenant_id),
            "citation_count": len(citations),
            "rag_metadata": response.get("metadata", {}),
            "citations": citations,
        },
    }


def _save_ragas_dataset(records: list[dict[str, Any]], output_dir: Path) -> Any:
    """把接口返回结果保存成 Ragas Dataset，方便后续追加和复盘。"""
    from ragas import Dataset

    dataset = Dataset(
        name=f"medical-rag-api-{time.strftime('%Y%m%d-%H%M%S')}",
        backend="local/csv",
        root_dir=str(output_dir),
    )
    for record in records:
        dataset.append(record)
    dataset.save()
    return dataset


def _build_judge_llm() -> Any:
    """构建 Ragas 评分用 LLM，默认复用 DashScope OpenAI-compatible 配置。"""
    from openai import OpenAI
    from ragas.llms import llm_factory

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，Ragas 指标无法调用评分模型")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    model = os.getenv("RAGAS_JUDGE_MODEL") or os.getenv("RAG_CHAT_MODEL", "qwen-plus")
    return llm_factory(model, provider="openai", client=client)


def _new_discrete_metric(discrete_metric_class: Any, judge_llm: Any, **kwargs: Any) -> Any:
    """兼容不同 Ragas 小版本的 DiscreteMetric 构造参数。"""
    try:
        return discrete_metric_class(**kwargs, llm=judge_llm)
    except TypeError:
        return discrete_metric_class(**kwargs)


def _build_metrics(judge_llm: Any) -> list[Any]:
    """定义两个和 RAG 质量直接相关的 Ragas 离散指标。"""
    from ragas.metrics import DiscreteMetric

    groundedness = _new_discrete_metric(
        DiscreteMetric,
        judge_llm,
        name="answer_groundedness",
        prompt=(
            "你是 RAG 评估员。请判断回答是否主要由检索上下文支持。\n"
            "问题：{user_input}\n"
            "回答：{response}\n"
            "检索上下文：{retrieved_contexts}\n"
            "如果回答中的关键结论能从上下文找到依据，返回 PASS；"
            "如果回答明显脱离上下文、编造依据或上下文为空却强行回答，返回 FAIL。"
        ),
        allowed_values=["PASS", "FAIL"],
    )
    clinical_correctness = _new_discrete_metric(
        DiscreteMetric,
        judge_llm,
        name="clinical_answer_quality",
        prompt=(
            "你是中医 RAG 评估员。请结合评分说明判断回答质量。\n"
            "问题：{user_input}\n"
            "评分说明：{grading_notes}\n"
            "回答：{response}\n"
            "检索上下文：{retrieved_contexts}\n"
            "如果回答符合评分说明，并且没有超出上下文作医疗承诺，返回 PASS；否则返回 FAIL。"
        ),
        allowed_values=["PASS", "FAIL"],
    )
    return [groundedness, clinical_correctness]


async def _score_metric(metric: Any, judge_llm: Any, sample: dict[str, Any]) -> Any:
    """兼容 Ragas metric 的同步/异步评分方法。

    中文：优先使用同步 score，因为脚本里的 Ragas judge 使用同步 OpenAI client。
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


def _score_value(score: Any) -> str:
    """提取 Ragas 评分值。"""
    if hasattr(score, "value"):
        return str(score.value)
    return str(score)


def _score_reason(score: Any) -> str:
    """提取 Ragas 评分理由；没有理由时返回空字符串。"""
    return str(getattr(score, "reason", "") or "")


async def _score_records(records: list[dict[str, Any]], output_dir: Path) -> Path:
    """逐条调用 Ragas 指标打分，并保存 CSV 结果。"""
    judge_llm = _build_judge_llm()
    metrics = _build_metrics(judge_llm)

    rows: list[dict[str, Any]] = []
    for record in records:
        contexts_text = "\n\n".join(
            f"[上下文 {index + 1}]\n{context}" for index, context in enumerate(record["retrieved_contexts"])
        )
        row: dict[str, Any] = {
            "user_input": record["user_input"],
            "response": record["response"],
            "grading_notes": record.get("grading_notes", ""),
            "citation_count": record["metadata"]["citation_count"],
            "duration_ms": record["metadata"]["duration_ms"],
        }
        for metric in metrics:
            score = await _score_metric(
                metric,
                judge_llm,
                {
                    "user_input": record["user_input"],
                    "response": record["response"],
                    "retrieved_contexts": contexts_text,
                    "grading_notes": record.get("grading_notes", ""),
                },
            )
            row[metric.name] = _score_value(score)
            row[f"{metric.name}_reason"] = _score_reason(score)
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"medical_ragas_scores_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_raw_records(records: list[dict[str, Any]], output_dir: Path) -> Path:
    """保存接口原始返回，便于排查 Ragas 分数背后的上下文和引用。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"medical_ragas_raw_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    _load_environment()
    parser = argparse.ArgumentParser(description="调用真实医疗 RAG 接口并使用 Ragas 评估回答质量")
    parser.add_argument("--api-base-url", default=os.getenv("RAGAS_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--tenant-id", default=os.getenv("RAGAS_TENANT_ID", "default"))
    parser.add_argument("--cases", default=os.getenv("RAGAS_EVAL_CASES_FILE"))
    parser.add_argument("--output-dir", default=os.getenv("RAGAS_EVAL_OUTPUT_DIR", "evals/experiments"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("RAGAS_API_TIMEOUT_SECONDS", "120")))
    args = parser.parse_args()

    api_key = os.getenv("RAG_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 RAG_API_KEY，无法调用受保护的医疗问答接口")

    cases = _load_cases(args.cases)
    records = [
        _query_medical_api(
            case=case,
            api_base_url=args.api_base_url,
            api_key=api_key,
            tenant_id=args.tenant_id,
            timeout=args.timeout,
        )
        for case in cases
    ]

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    raw_path = _save_raw_records(records, output_dir)
    print(f"已保存真实接口返回: {raw_path}")

    try:
        dataset = _save_ragas_dataset(records, output_dir)
        print(f"已保存 Ragas Dataset: {dataset.name}")
    except Exception as exc:
        print(f"跳过 Ragas Dataset 落盘，但继续执行指标评分: {exc}")

    score_path = asyncio.run(_score_records(records, output_dir))
    print(f"已保存 Ragas 评估结果: {score_path}")


if __name__ == "__main__":
    main()
