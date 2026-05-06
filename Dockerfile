FROM python:3.11-slim
# PYTHONDONTWRITEBYTECODE=1  表示不生成 .pyc 缓存文件。
ENV PYTHONDONTWRITEBYTECODE=1
# PYTHONUNBUFFERED=1 表示日志直接输出，不缓存，容器里看日志更及时。
ENV PYTHONUNBUFFERED=1
# 容器里的工作目录设为 /app
WORKDIR /app
# 创建一个普通用户 appuser，后面不用 root 用户跑服务，安全一点。
RUN useradd --create-home --shell /bin/bash appuser

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "codex55_rag_project.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Dockerfile说明：告诉docker怎么构建镜像




