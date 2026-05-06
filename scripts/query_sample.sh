#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${RAG_API_KEY:-change-me}"

curl -sS -X POST "${BASE_URL}/v1/query" \
  -H "content-type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d '{
    "tenant_id": "tenant-a",
    "question": "生产级 RAG 应该包含哪些模块？",
    "metadata_filter": {
      "category": "architecture"
    }
  }'

