# 生产级 RAG 架构说明

## 模块边界

- `api`：FastAPI 入口，负责 `AppState` 启动装配、鉴权、请求校验、响应模型、请求日志。
- `config`：环境变量配置。
- `core`：业务模型、端口接口、`RagPipeline` 与 LangChain 索引编排。
- `services`：LangChain Embedding、LLM、Retriever、Reranker、Prompt、Chunker 等 RAG 组件。
- `data/vector_store`：LangChain PGVector 生产实现、旧 pgvector 对照实现和本地内存向量库。
- `loaders`：文档加载器。
- `monitoring`：结构化 JSON 日志、request id、耗时记录。
- `security`：API Key 鉴权。

## 启动流程

```text
FastAPI lifespan
  -> build_app_state(settings)
  -> create LangChain Embeddings / PGVector / Retriever / Reranker / LLM / RagPipeline
  -> init pgvector extension and LangChain collection
  -> set_app_state(state)
  -> requests reuse get_app_state()
  -> close VectorStore on shutdown
```

## 请求路径

```text
POST /v1/ingest
  -> validate tenant/documents
  -> LangChain text splitter
  -> LangChain PGVector add_documents
  -> DashScope text-embedding-v4
  -> Postgres + pgvector collection with tenant_id + metadata

POST /v1/query
  -> validate tenant/question/filter
  -> LangChain PGVector similarity search
  -> DashScope text-embedding-v4 query embedding
  -> tenant-isolated metadata filter
  -> DashScope rerank
  -> prompt with citations
  -> LangChain ChatOpenAI via DashScope compatible endpoint
  -> answer + citations
```

## 生产关注点

- 多租户隔离：所有 chunk 写入 `tenant_id`，查询必须带 `tenant_id`。
- 权限过滤：通过 `metadata_filter` 下推到 Postgres JSONB。
- 可观测性：日志输出 JSON，包含 request id 与 duration。
- 可替换性：真实 provider 都在 `services` 和 `data/vector_store`，核心 pipeline 不依赖具体供应商。
- 部署：提供 Dockerfile 与 pgvector docker-compose。

## 第一版选型理由

- 选择 LangChain：复用成熟 RAG 组件和 PGVector 集成，减少自研检索代码风险。
- 选择 pgvector：减少独立向量数据库运维成本，同时保留 SQL、JSONB filter 和事务能力。
- 选择阿里云 OpenAI-compatible：通过 DashScope 兼容模式统一接入 chat 和 embedding。
- 选择 API Key：适合第一版服务间调用；权限细粒度控制由 `tenant_id` 和 `metadata_filter` 承担。
- 选择同步索引：工程路径短，调用方能明确知道入库完成；超大批量导入再升级为异步任务。
