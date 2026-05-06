FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir \
        -i https://mirrors.aliyun.com/pypi/simple \
        --trusted-host mirrors.aliyun.com \
        --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        -i https://mirrors.aliyun.com/pypi/simple \
        --trusted-host mirrors.aliyun.com \
        --no-build-isolation \
        .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "codex55_rag_project.main:app", "--host", "0.0.0.0", "--port", "8000"]
