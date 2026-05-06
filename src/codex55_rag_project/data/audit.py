"""Database-backed audit records for medical RAG requests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


VALID_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass(frozen=True)
class MedicalQueryAuditRecord:
    request_id: str
    tenant_id: str
    question: str
    rewritten_queries: list[str]
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    answer: str
    duration_ms: float
    error: str | None = None


class MedicalAuditStore:
    """Stores query audit records in Postgres."""

    def __init__(self, connection: str, table_name: str) -> None:
        if not VALID_IDENTIFIER.match(table_name):
            raise ValueError("table_name must be a safe SQL identifier")
        self.connection = _normalize_psycopg_connection(connection)
        self.table_name = table_name

    def init_schema(self) -> None:
        import psycopg

        with psycopg.connect(self.connection) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    request_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    rewritten_queries JSONB NOT NULL,
                    retrieved_chunks JSONB NOT NULL,
                    reranked_chunks JSONB NOT NULL,
                    answer TEXT NOT NULL,
                    duration_ms DOUBLE PRECISION NOT NULL,
                    error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table_name}_tenant_created_idx "
                f"ON {self.table_name} (tenant_id, created_at DESC)"
            )
            conn.commit()

    def write_query(self, record: MedicalQueryAuditRecord) -> None:
        import psycopg

        with psycopg.connect(self.connection) as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table_name}
                    (
                        request_id,
                        tenant_id,
                        question,
                        rewritten_queries,
                        retrieved_chunks,
                        reranked_chunks,
                        answer,
                        duration_ms,
                        error
                    )
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s)
                ON CONFLICT (request_id) DO UPDATE SET
                    rewritten_queries = EXCLUDED.rewritten_queries,
                    retrieved_chunks = EXCLUDED.retrieved_chunks,
                    reranked_chunks = EXCLUDED.reranked_chunks,
                    answer = EXCLUDED.answer,
                    duration_ms = EXCLUDED.duration_ms,
                    error = EXCLUDED.error
                """,
                (
                    record.request_id,
                    record.tenant_id,
                    record.question,
                    json.dumps(record.rewritten_queries, ensure_ascii=False),
                    json.dumps(record.retrieved_chunks, ensure_ascii=False),
                    json.dumps(record.reranked_chunks, ensure_ascii=False),
                    record.answer,
                    record.duration_ms,
                    record.error,
                ),
            )
            conn.commit()


def _normalize_psycopg_connection(connection: str) -> str:
    if connection.startswith("jdbc:postgresql://"):
        return "postgresql://" + connection.removeprefix("jdbc:postgresql://")
    return connection.replace("postgresql+psycopg://", "postgresql://", 1)
