#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
RAG_API_KEY="${RAG_API_KEY:-change-me}"
TENANT_ID="${TENANT_ID:-default}"
PDF_PATH="${PDF_PATH:-中医临床诊疗智能助手.pdf}"

curl -sS -X POST "${BASE_URL}/v1/admin/pdf-ingestions" \
  -H "content-type: application/json" \
  -H "x-api-key: ${RAG_API_KEY}" \
  -d "{\"tenant_id\":\"${TENANT_ID}\",\"pdf_path\":\"${PDF_PATH}\"}"
