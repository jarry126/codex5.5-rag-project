"""统一日志模块。

中文：输出单行可读日志，并通过 ContextVar 自动携带 request_id，方便本地调试和线上排查。
English: Outputs readable single-line logs and propagates request_id through ContextVar.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> None:
    """设置当前请求链路 ID。"""
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    """获取当前请求链路 ID。"""
    return _request_id_var.get()


def clear_request_id() -> None:
    """清理当前请求链路 ID。"""
    _request_id_var.set(None)


class RequestIdFilter(logging.Filter):
    """给每条日志自动补充 request_id。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(record, "request_id", None) or get_request_id() or "-"
        return True


class ReadableFormatter(logging.Formatter):
    """单行可读日志格式化器。"""

    extra_fields = (
        "event",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "tenant_id",
        "query_count",
        "candidate_count",
        "final_count",
        "ingestion_id",
        "documents",
        "pages",
        "chunks",
        "case_count",
    )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = []
        for key in self.extra_fields:
            value = getattr(record, key, None)
            if value is not None:
                fields.append(f"{key}={value}")
        if fields:
            base = f"{base} | {' '.join(fields)}"
        return base


def configure_logging(level: str) -> None:
    """配置全局日志。

    中文：统一输出可读单行日志，避免本地和生产两套格式割裂。
    English: Uses one readable log format for both local and production.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        ReadableFormatter(
            "%(asctime)s %(levelname)s [%(request_id)s] [%(name)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _configure_third_party_loggers()


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


def _configure_third_party_loggers() -> None:
    """降低第三方库噪声，让业务日志更容易看。"""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langsmith.client").setLevel(logging.ERROR)
