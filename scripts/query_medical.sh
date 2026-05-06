#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
RAG_API_KEY="${RAG_API_KEY:-change-me}"
TENANT_ID="${TENANT_ID:-default}"
QUESTION="${1:-失眠应该如何辨证调理？}"

curl -sS -X POST "${BASE_URL}/v1/medical/query" \
  -H "content-type: application/json" \
  -H "x-api-key: ${RAG_API_KEY}" \
  -d "{\"tenant_id\":\"${TENANT_ID}\",\"question\":\"${QUESTION}\"}"
