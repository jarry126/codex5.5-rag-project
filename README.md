# Codex5.5 RAG Project

这是一个面向生产的 RAG 服务代码。生产链路基于 LangChain，使用 Postgres + pgvector 做持久化向量检索，使用阿里云百炼 DashScope 的 OpenAI-compatible API 接入大模型、Embedding 和重排序。

## 架构

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

## 真实生产组件

- API: FastAPI，通用接口在 `/v1/ingest` 和 `/v1/query`，医疗 RAG 接口在 `/v1/admin/pdf-ingestions` 和 `/v1/medical/query`。
- LangChain: 使用 `langchain-openai`、`langchain-postgres`、`langchain-text-splitters`。
- Embedding: 阿里云百炼 OpenAI-compatible embedding，默认 `text-embedding-v4`，1024 维。
- Vector DB: 通用链路使用 LangChain `PGVector`；医疗链路使用新版 `PGVectorStore`，支持 hybrid search 和 JSONB metadata filter。
- LLM: LangChain `ChatOpenAI` 指向阿里云百炼兼容接口，默认 `qwen-plus`。
- Reranker: 阿里云重排序接口，默认 `qwen3-rerank`。
- Auth: `x-api-key` 简单服务级鉴权，可替换为 OAuth/JWT/网关鉴权。
- Observability: JSON 日志、request id、耗时字段。

## 启动装配

FastAPI 启动时会在 `lifespan` 中创建 `AppState`，一次性初始化 Embedding、VectorStore、Retriever、LLM 和 `RagPipeline`。请求处理时通过 `get_app_state()` 复用这些对象，不会每次请求重新创建模型客户端或数据库连接池。

## 本地启动

```bash
cp .env.example .env
# 修改 .env 里的 DASHSCOPE_API_KEY、RAG_POSTGRES_DSN、RAG_MEDICAL_POSTGRES_DSN 和 RAG_API_KEY
docker compose -f deploy/docker-compose.yml up --build
```

## 调用示例

```bash
bash scripts/ingest_sample.sh
bash scripts/query_sample.sh
```

## 医疗 PDF RAG

生产医疗链路使用独立数据库 `rag_medical`，避免污染已有 `postgres` 数据库。先初始化数据库和 pgvector 扩展：

```bash
PGPASSWORD=your-password bash scripts/create_medical_db.sh
```

启动 API 后导入默认 PDF：

```bash
bash scripts/ingest_medical_pdf.sh
```

调用医疗问答接口：

```bash
bash scripts/query_medical.sh "失眠应该如何辨证调理？"
```

医疗问答会执行问题改写、多路 hybrid 召回、去重融合、重排序和 LLM 生成，并将链路摘要写入 JSON 日志和 `medical_rag_audit` 审计表。

## 本地无外部依赖测试

```bash
python3 examples/local_rag_demo.py
python3 -m pytest
```

## 文件说明

- `src/codex55_rag_project/api`: FastAPI HTTP 入口、`AppState` 启动装配、请求响应模型。
- `src/codex55_rag_project/config`: 环境变量配置。
- `src/codex55_rag_project/core`: 业务模型、接口边界、RAG 查询编排和文档索引编排。
- `src/codex55_rag_project/services`: Embedding、LLM、Retriever、Reranker、Prompt、Chunker 等 RAG 组件。
- `src/codex55_rag_project/data/vector_store`: LangChain PGVector、旧 pgvector 对照实现和本地内存向量库。
- `src/codex55_rag_project/loaders`: 文档加载器。
- `src/codex55_rag_project/monitoring`: JSON 日志、request id 和耗时记录。
- `src/codex55_rag_project/security`: API Key 鉴权。
- `src/codex55_rag_project/bootstrap`: 本地 demo 和测试用的组装入口。
- `deploy`: Docker Compose 与数据库初始化。

更完整的架构说明见 `docs_architecture.md`。

Linux 生产部署步骤见 `docs/production_deploy_linux.md`。

## 技术选型

- `Python + FastAPI`: RAG/LLM 生态成熟，便于对接模型、向量库和评测工具。
- `LangChain + Postgres + pgvector`: 复用 LangChain 生态，同时让本地已有 Postgres 可以直接作为向量库。
- `阿里云百炼 OpenAI-compatible API`: 使用 `https://dashscope.aliyuncs.com/compatible-mode/v1` 接入 chat/embedding。
- `API Key`: 第一版服务间鉴权简单可靠；后续可替换为 JWT、OAuth 或网关鉴权。
- 同步索引: 请求返回即完成切分、向量化和入库；大批量任务后续再扩展队列。

## 本地接入阿里云和已有 Postgres

`.env` 至少需要配置：

```bash
DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_POSTGRES_DSN=postgresql+psycopg://user:password@localhost:5432/dbname
RAG_MEDICAL_POSTGRES_DSN=postgresql+psycopg://user:password@localhost:5432/rag_medical
RAG_API_KEY=change-me
```

注意：不要把真实 API Key 提交到代码仓库。
