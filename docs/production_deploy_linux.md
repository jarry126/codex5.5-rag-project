# Linux 生产部署说明

本文档用于把当前 RAG 服务部署到 Linux 生产环境。生产环境不要把 `.env`、数据库密码、DashScope API Key 打进镜像，统一通过运行时环境变量、部署平台 Secret 或服务器本地 `.env` 注入。

## 1. 生产架构建议

推荐拆成两层：

```text
Nginx / API Gateway
  -> FastAPI RAG API 容器
  -> PostgreSQL + pgvector + pg_jieba
  -> DashScope / OpenAI-compatible API
```

生产建议：

- API 服务使用 Docker 镜像运行。
- PostgreSQL 可以使用公司已有数据库，也可以独立部署，但必须安装 `pgvector` 和 `pg_jieba`。
- 医疗 RAG 使用独立数据库 `rag_medical`，不要和默认 `postgres` 业务库混用。
- `.env` 只放在服务器或部署平台 Secret 中，不提交代码仓库。
- 外部访问建议走 Nginx/网关，API 容器只暴露内网端口。

## 2. Linux 服务器准备

安装基础组件：

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin postgresql-client curl
sudo systemctl enable --now docker
```

如果数据库也部署在这台机器上，需要准备 PostgreSQL，并安装扩展：

```text
PostgreSQL
pgvector
pg_jieba
```

注意：`pgvector/pgvector` 官方镜像通常只包含 `pgvector`，不一定包含 `pg_jieba`。当前项目医疗混合检索默认使用 `jiebacfg`，所以生产库必须先执行成功：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_jieba;
```

并确认存在中文分词配置：

```sql
SELECT cfgname FROM pg_ts_config WHERE cfgname LIKE 'jieba%';
```

应至少看到：

```text
jiebacfg
jiebahmm
jiebamp
jiebaqry
```

## 3. 初始化数据库

使用维护库创建医疗库：

```bash
export DB_HOST=127.0.0.1
export DB_PORT=5432
export DB_ADMIN_DB=postgres
export DB_USER=your_db_user
export DB_NAME=rag_medical
export PGPASSWORD='your_db_password'

bash scripts/create_medical_db.sh
```

脚本会做两件事：

```sql
CREATE DATABASE rag_medical;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_jieba;
```

如果 `pg_jieba` 报错，说明数据库服务器还没有安装 pg_jieba 扩展文件，需要先在数据库机器上安装扩展，再重新执行脚本。

## 4. 生产环境变量

在服务器上创建 `.env`，或者把这些变量配置到部署平台 Secret：

```bash
DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

RAG_API_KEY=replace-with-long-random-secret

RAG_POSTGRES_DSN=postgresql+psycopg://user:password@postgres-host:5432/rag
RAG_MEDICAL_POSTGRES_DSN=postgresql+psycopg://user:password@postgres-host:5432/rag_medical
RAG_MEDICAL_ADMIN_POSTGRES_DSN=postgresql://user:password@postgres-host:5432/postgres

RAG_MEDICAL_DATABASE_NAME=rag_medical
RAG_MEDICAL_VECTOR_TABLE=medical_rag_chunks
RAG_MEDICAL_AUDIT_TABLE=medical_rag_audit
RAG_MEDICAL_DEFAULT_PDF_PATH=中医临床诊疗智能助手.pdf
RAG_MEDICAL_TEXT_SEARCH_CONFIG=jiebacfg
RAG_MEDICAL_QUERY_REWRITE_COUNT=3

RAG_CHAT_MODEL=qwen-plus
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_VECTOR_DIMENSIONS=1024
RAG_RERANK_ENABLED=true
RAG_RERANK_MODEL=qwen3-rerank

RAG_CHUNK_SIZE=900
RAG_CHUNK_OVERLAP=150
RAG_CANDIDATE_K=24
RAG_FINAL_K=6
RAG_MIN_SCORE=0.0
```

生产建议：

- `RAG_API_KEY` 使用长随机字符串。
- 数据库账号使用最小权限账号，不要使用超级管理员账号跑 API。
- 不要在日志、镜像、Git 仓库中保存真实密码和 API Key。

## 5. 构建 API 镜像

在项目根目录执行：

```bash
docker build -t codex55-rag-api:0.1.0 .
```

确认镜像：

```bash
docker images | grep codex55-rag-api
```

## 6. 启动 API 容器

如果 `.env` 在项目根目录：

```bash
docker run -d \
  --name codex55-rag-api \
  --restart unless-stopped \
  --env-file .env \
  -p 8000:8000 \
  codex55-rag-api:0.1.0
```

查看日志：

```bash
docker logs -f codex55-rag-api
```

健康检查：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

Swagger：

```text
http://服务器IP:8000/docs
```

生产环境建议不要直接公网暴露 Swagger，可以通过 Nginx、VPN、内网或鉴权网关控制访问。

## 7. 导入医疗 PDF

把 PDF 放到服务器项目目录，或者挂载到容器内可访问路径。

如果 PDF 在项目根目录并且容器内没有这个文件，需要启动容器时挂载：

```bash
docker run -d \
  --name codex55-rag-api \
  --restart unless-stopped \
  --env-file .env \
  -v "$(pwd)/中医临床诊疗智能助手.pdf:/app/中医临床诊疗智能助手.pdf:ro" \
  -p 8000:8000 \
  codex55-rag-api:0.1.0
```

调用入库接口：

```bash
export BASE_URL=http://127.0.0.1:8000
export RAG_API_KEY=replace-with-long-random-secret
export TENANT_ID=default
export PDF_PATH=中医临床诊疗智能助手.pdf

bash scripts/ingest_medical_pdf.sh
```

入库成功后，数据库里应有数据：

```sql
SELECT count(*) FROM medical_rag_chunks;
SELECT count(*) FROM medical_rag_audit;
SELECT to_tsvector('jiebacfg', '中医临床诊疗智能助手');
```

## 8. 问答验证

调用医疗问答：

```bash
export BASE_URL=http://127.0.0.1:8000
export RAG_API_KEY=replace-with-long-random-secret

bash scripts/query_medical.sh "失眠应该如何辨证调理？"
```

返回结果应包含：

```text
answer
citations / contexts
metadata.request_id
metadata.rewritten_queries
metadata.retrieval_mode = hybrid_multi_query
metadata.candidate_count
metadata.final_count
```

同时审计表会记录本次问答：

```sql
SELECT request_id, tenant_id, question, duration_ms, error
FROM medical_rag_audit
ORDER BY created_at DESC
LIMIT 5;
```

## 9. Nginx 反向代理示例

```nginx
server {
    listen 80;
    server_name your-domain.example.com;

    client_max_body_size 20m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

生产建议启用 HTTPS，并把鉴权、限流、访问日志接入公司网关或 Nginx。

## 10. 发布和回滚

发布新版本：

```bash
docker build -t codex55-rag-api:0.1.1 .
docker stop codex55-rag-api
docker rm codex55-rag-api
docker run -d \
  --name codex55-rag-api \
  --restart unless-stopped \
  --env-file .env \
  -p 8000:8000 \
  codex55-rag-api:0.1.1
```

回滚旧版本：

```bash
docker stop codex55-rag-api
docker rm codex55-rag-api
docker run -d \
  --name codex55-rag-api \
  --restart unless-stopped \
  --env-file .env \
  -p 8000:8000 \
  codex55-rag-api:0.1.0
```

## 11. 上线前检查清单

- `.env` 没有提交到 Git。
- API 镜像构建成功。
- 数据库能连通。
- `vector` 扩展已启用。
- `pg_jieba` 扩展已启用。
- `jiebacfg` 能查询到。
- `/healthz` 返回正常。
- `/readyz` 返回正常。
- PDF 入库成功。
- `/v1/medical/query` 能返回答案和 citations。
- `medical_rag_audit` 有审计记录。
- 日志中没有输出数据库密码和 API Key。
- Nginx/网关已配置 HTTPS、限流和访问控制。
