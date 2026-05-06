#!/usr/bin/env bash
set -euo pipefail

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_ADMIN_DB="${DB_ADMIN_DB:-postgres}"
DB_USER="${DB_USER:-liushanshan}"
DB_NAME="${DB_NAME:-rag_medical}"

psql "postgresql://${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_ADMIN_DB}" \
  -v ON_ERROR_STOP=1 \
  -v db_name="${DB_NAME}" <<'SQL'
SELECT format('CREATE DATABASE %I', :'db_name')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'db_name')\gexec
SQL

psql "postgresql://${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}" \
  -v ON_ERROR_STOP=1 <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_jieba;
SQL
