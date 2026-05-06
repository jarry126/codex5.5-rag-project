#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${RAG_API_KEY:-change-me}"

curl -sS -X POST "${BASE_URL}/v1/ingest" \
  -H "content-type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d '{
    "tenant_id": "tenant-a",
    "documents": [
      {
        "id": "rag-architecture",
        "text": "生产级 RAG 包含数据接入、切分、Embedding、向量索引、召回、重排、Prompt 构造、LLM 生成、引用返回、权限过滤、观测和评测闭环。",
        "metadata": {
          "source": "sample",
          "category": "architecture"
        }
      }
    ]
  }'

