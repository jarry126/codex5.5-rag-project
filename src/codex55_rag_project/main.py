import sys
import asyncio
import os

import uvicorn

from codex55_rag_project.api.app import app

__all__ = ["app"]


"""统一执行入口。

可用于本地测试启动：

    python -m codex55_rag_project.main

也可以通过环境变量覆盖监听地址和端口：

    RAG_HOST=0.0.0.0 RAG_PORT=10222 python -m codex55_rag_project.main
"""


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


if __name__ == "__main__":
    uvicorn.run(
        "codex55_rag_project.main:app",
        host=os.getenv("RAG_HOST", "127.0.0.1"),
        port=int(os.getenv("RAG_PORT", "9009")),
    )
