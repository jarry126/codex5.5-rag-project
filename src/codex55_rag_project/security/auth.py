"""API Key 认证中间件。

中文：通过 x-api-key header 验证请求身份；生产环境应替换为 JWT 或 OAuth。
English: Authenticates requests via x-api-key header; replace with JWT/OAuth for production.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from codex55_rag_project.config import get_settings


def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """验证 API Key。

    中文：如果配置了 api_key 且请求 header 不匹配，返回 401。
    English: Returns 401 if api_key is configured and request header doesn't match.
    """
    expected = get_settings().api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")

