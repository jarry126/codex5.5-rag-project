"""JSON 结构化日志模块。

中文：输出 JSON 格式日志，便于日志平台解析和检索；包含 request_id、tenant_id、duration 等字段。
English: Outputs JSON format logs for easy parsing by log platforms; includes request_id, tenant_id, duration fields.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator


class JsonFormatter(logging.Formatter):
    """JSON 格式化器。

    中文：将日志记录转换为 JSON 对象，包含时间戳、级别、logger 名、消息和额外字段。
    English: Converts log records to JSON object with timestamp, level, logger name, message, and extra fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为 JSON。"""
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 添加常用业务字段；避免把完整请求体或密钥写进日志。
        for key in (
            "request_id",
            "tenant_id",
            "duration_ms",
            "event",
            "query_count",
            "candidate_count",
            "final_count",
            "ingestion_id",
            "documents",
            "pages",
            "chunks",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        # 添加异常信息
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> None:
    """配置全局日志。

    中文：设置 JSON 格式化器，输出到 stdout，便于容器日志收集。
    English: Sets JSON formatter, outputs to stdout for container log collection.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """获取 logger 实例。"""
    return logging.getLogger(name)


@contextmanager
def log_duration(logger: logging.Logger, event: str, **fields: object) -> Iterator[None]:
    """记录操作耗时。

    中文：在 context manager 开始时计时，结束时记录 duration_ms。
    English: Starts timing at context manager entry, logs duration_ms at exit.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(event, extra={"event": event, "duration_ms": duration_ms, **fields})
