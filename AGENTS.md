# AGENTS.md

## Project Overview

This repository is a production-oriented Python RAG service generated for Codex 5.5 evaluation/demo work.

The service exposes a FastAPI API for document ingestion and question answering. The production path uses LangChain, Postgres + pgvector, and Alibaba Cloud DashScope through OpenAI-compatible APIs.

Main request flow:

```text
Documents
  -> Loader
  -> Chunker
  -> Embedder
  -> VectorStore
  -> Retriever
  -> Optional Reranker
  -> PromptBuilder
  -> LLM
  -> Answer + Citations
```

## Important Paths

- `src/codex55_rag_project/api`: FastAPI app, request schemas, dependency wiring, app state.
- `src/codex55_rag_project/config`: Pydantic settings and environment variable mapping.
- `src/codex55_rag_project/core`: Domain models, ports, ingestion pipeline, RAG orchestration.
- `src/codex55_rag_project/services`: Providers for chunking, embedding, retrieval, reranking, prompting, and LLM calls.
- `src/codex55_rag_project/data/vector_store`: LangChain PGVector, direct pgvector reference implementation, and in-memory store.
- `src/codex55_rag_project/loaders`: Document loaders.
- `src/codex55_rag_project/monitoring`: JSON logging, request id, and duration logging helpers.
- `src/codex55_rag_project/security`: API key authentication.
- `tests`: Unit tests for API behavior, settings, pgvector filtering, and RAG pipeline behavior.
- `deploy`: Docker Compose and database initialization.
- `scripts`: Sample ingest/query curl wrappers.

## Common Commands

Run all tests:

```bash
python3 -m pytest
```

Run local dependency-free demo:

```bash
python3 examples/local_rag_demo.py
```

Start the production-like stack:

```bash
cp .env .env
docker compose -f deploy/docker-compose.yml up --build
```

Then configure `.env` with at least:

```bash
DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_POSTGRES_DSN=postgresql+psycopg://user:password@localhost:5432/dbname
RAG_API_KEY=change-me
```

Sample API calls:

```bash
bash scripts/ingest_sample.sh
bash scripts/query_sample.sh
```

## Development Notes

- Keep `core` provider-agnostic. It should depend on ports/interfaces, not concrete SDKs or database clients.
- Put vendor-specific integrations under `services` or `data/vector_store`.
- Preserve tenant isolation. `tenant_id` must flow from API requests through retrieval/vector-store filters and should not be applied only after retrieval.
- Do not commit real API keys or secrets. Use `.env` locally.
- The default production model path is DashScope/OpenAI-compatible:
  - Chat model default: `qwen-plus`
  - Embedding model default: `text-embedding-v4`
  - Reranker default: `qwen3-rerank`
- The local tests are designed to run without external services.

## API Surface

- `GET /healthz`: Liveness check, no external dependency access.
- `GET /readyz`: Readiness check, validates database/vector-store connectivity.
- `POST /v1/ingest`: Synchronously chunks, embeds, and stores documents.
- `POST /v1/query`: Retrieves tenant-filtered contexts, optionally reranks, builds a citation prompt, and returns an answer with citations.

Protected endpoints use `x-api-key`.

## Testing Expectations

Before handing off code changes, run:

```bash
python3 -m pytest
```

Add or update tests when changing:

- API request/response schemas.
- Settings or environment variable aliases.
- Tenant filtering or metadata filter behavior.
- Retrieval, reranking, prompt construction, or citation behavior.
- Provider wiring in `api/dependencies.py`.

## Repository State Notes

At the time this file was created, the directory was not a Git repository. Avoid relying on `git status` as a source of truth unless a `.git` directory is later initialized.
