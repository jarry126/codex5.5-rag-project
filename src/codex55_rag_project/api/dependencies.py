from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, status

from codex55_rag_project.config.settings import PROJECT_ROOT
from codex55_rag_project.config import Settings
from codex55_rag_project.core.medical_pipeline import MedicalRagPipeline, MedicalRetrievalConfig
from codex55_rag_project.core.langchain_ingestion import LangChainIngestionPipeline
from codex55_rag_project.core.pipeline import RagPipeline, RetrievalConfig
from codex55_rag_project.data.audit import MedicalAuditStore
from codex55_rag_project.data.vector_store.hybrid_pgvector import HybridPgVectorStore
from codex55_rag_project.data.vector_store.langchain_pgvector import LangChainPgVectorStore
from codex55_rag_project.loaders.pdf import PdfDocumentLoader
from codex55_rag_project.loaders.text import StaticDocumentLoader
from codex55_rag_project.services.chunking import RecursiveTextChunker
from codex55_rag_project.services.embedding.langchain_dashscope import build_dashscope_embeddings
from codex55_rag_project.services.llm.langchain_dashscope import LangChainDashScopeChatLLM
from codex55_rag_project.services.llm.query_rewriter import DashScopeQueryRewriter
from codex55_rag_project.services.prompt.citation import CitationPromptBuilder
from codex55_rag_project.services.reranker.dashscope import DashScopeReranker
from codex55_rag_project.services.reranker.keyword import KeywordOverlapReranker
from codex55_rag_project.services.retriever.langchain_pgvector import LangChainPgVectorRetriever
from codex55_rag_project.services.retriever.multi_query import MultiQueryRetriever


@dataclass(frozen=True)
class PdfIngestionStats:
    ingestion_id: str
    documents: int
    pages: int
    chunks: int
    source: str


@dataclass
class AppState:
    """应用启动后创建好的 RAG 服务状态。

    这里的对象会被所有请求复用，避免每次请求都重新创建模型客户端、数据库连接池和 pipeline。
    """

    settings: Settings
    embeddings: object
    vector_store: LangChainPgVectorStore
    medical_vector_store: HybridPgVectorStore
    medical_audit_store: MedicalAuditStore
    chunker: RecursiveTextChunker
    retriever: LangChainPgVectorRetriever
    llm: LangChainDashScopeChatLLM
    pipeline: RagPipeline
    medical_pipeline: MedicalRagPipeline

    def init_schema(self) -> None:
        # 启动时初始化 pgvector 表和索引，保证服务 ready 后就能读写向量数据。
        self.vector_store.init_schema()
        self.medical_vector_store.init_schema()
        self.medical_audit_store.init_schema()

    def health_check(self) -> None:
        # readyz 使用它检查数据库是否可用；healthz 则只检查进程是否存活。
        self.vector_store.health_check()
        self.medical_vector_store.health_check()

    def close(self) -> None:
        # 应用关闭时释放数据库连接池，避免容器滚动发布时残留连接。
        self.vector_store.close()
        self.medical_vector_store.close()

    def build_ingestion(self, loader: StaticDocumentLoader) -> LangChainIngestionPipeline:
        # 每次 ingest 请求的文档不同，所以 loader 按请求创建；chunker/vector_store 复用启动态对象。
        return LangChainIngestionPipeline(loader, self.chunker, self.vector_store)


    """
        处理文档，没有特殊的东西，需要注意：将document保存到postgresql，默认会在数据库中增加content_csv字段（因为库中使用了jieba分词。）
    """
    def ingest_pdf(self, pdf_path: str | None, tenant_id: str) -> PdfIngestionStats:
        """处理 PDF 文档入库。

        中文：生成唯一的 ingestion_id 用于追踪，加载 PDF 并按页切分，自动向量化后存入医疗向量库。
        English: Generates unique ingestion_id for tracking, loads PDF, splits by page, auto-embeds and stores in medical vector store.

        Args:
            pdf_path: PDF 文件路径，如果为空则使用配置的默认路径
            tenant_id: 租户 ID，用于多租户隔离

        Returns:
            PdfIngestionStats: 包含入库统计信息（文档数、页数、切片数等）
        """
        # 生成唯一的入库 ID，用于审计日志和问题追踪
        ingestion_id = str(uuid4())
        # 确定 PDF 源文件路径：绝对路径原样使用；相对路径统一按项目根目录解析，避免 PyCharm 工作目录变化导致找不到文件。
        source_path = _resolve_project_path(pdf_path or self.settings.medical_default_pdf_path)
        source = str(source_path)
        # 创建 PDF 加载器，按页读取并附加租户和入库 ID 元数据
        loader = PdfDocumentLoader(source, tenant_id=tenant_id, ingestion_id=ingestion_id)
        # 构建 LangChain 索引流水线：loader 读取 → chunker 切分 → medical_vector_store 自动向量化入库
        pipeline = LangChainIngestionPipeline(loader, self.chunker, self.medical_vector_store)
        # 执行索引流程，返回统计信息（文档数、切片数）
        stats = pipeline.run()
        return PdfIngestionStats(
            ingestion_id=ingestion_id,  # 本次入库的唯一标识
            documents=stats.documents,  # 处理的文档数量（PDF 页数）
            pages=stats.documents,  # PDF 页数（与 documents 相同，因为每页是一个 Document）
            chunks=stats.chunks,  # 切分后的 chunk 总数
            source=source,  # 源文件路径
        )


app_state: AppState | None = None  # 这个全局变量，一开始给的是None


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def build_app_state(settings: Settings) -> AppState:
    """按配置组装生产版 RAG 服务。

    这个函数相当于 3 号项目 lifespan 里的显式初始化逻辑，只是抽成函数方便测试。
    """
    if not settings.openai_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY or OPENAI_API_KEY is required for production providers")

    # Embedding 使用 LangChain OpenAIEmbeddings 指向阿里云百炼 OpenAI-compatible endpoint。
    embeddings = build_dashscope_embeddings(settings)
    # 生产向量库使用 LangChain PGVector，collection schema 由 langchain-postgres 管理。
    vector_store = LangChainPgVectorStore( # LangChainPgVectorStore是自建的类，里面用的是PgVector
        connection=settings.postgres_dsn,
        collection_name=settings.vector_collection_name,
        embeddings=embeddings,
        embedding_length=settings.vector_dimensions,
    )
    medical_vector_store = HybridPgVectorStore(
        connection=settings.medical_postgres_dsn,
        table_name=settings.medical_vector_table,
        embeddings=embeddings,
        embedding_length=settings.vector_dimensions,
        text_search_config=settings.medical_text_search_config,
    )
    medical_audit_store = MedicalAuditStore(
        connection=settings.medical_postgres_dsn,
        table_name=settings.medical_audit_table,
    )
    # chunker/retriever/reranker/prompt_builder/llm 都通过接口接入 pipeline，后续可单独替换。
    chunker = RecursiveTextChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    retriever = LangChainPgVectorRetriever(vector_store)
    llm = LangChainDashScopeChatLLM(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        timeout=settings.request_timeout_seconds,
    )
    reranker = (
        DashScopeReranker(
            api_key=settings.openai_api_key,
            base_url=settings.rerank_base_url,
            model=settings.rerank_model,
            timeout=settings.request_timeout_seconds,
        )
        if settings.rerank_enabled
        else KeywordOverlapReranker()  # 本地写的重排序器
    )
    query_rewriter = DashScopeQueryRewriter(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
        temperature=0.0,
        timeout=settings.request_timeout_seconds,
    )
    pipeline = RagPipeline(
        retriever=retriever,
        reranker=reranker,
        prompt_builder=CitationPromptBuilder(),
        llm=llm,
        config=RetrievalConfig(
            candidate_k=settings.candidate_k,
            final_k=settings.final_k,
            min_score=settings.min_score,
        ),
    )
    medical_pipeline = MedicalRagPipeline(
        query_rewriter=query_rewriter,
        retriever=MultiQueryRetriever(LangChainPgVectorRetriever(medical_vector_store)),
        reranker=reranker,
        prompt_builder=CitationPromptBuilder(),
        llm=llm,
        audit_store=medical_audit_store,
        config=MedicalRetrievalConfig(
            candidate_k=settings.candidate_k,
            final_k=settings.final_k,
            min_score=settings.min_score,
            rewrite_count=settings.medical_query_rewrite_count,
        ),
    )

    return AppState(
        settings=settings,
        embeddings=embeddings,
        vector_store=vector_store,
        medical_vector_store=medical_vector_store,
        medical_audit_store=medical_audit_store,
        chunker=chunker,
        retriever=retriever,
        llm=llm,
        pipeline=pipeline,
        medical_pipeline=medical_pipeline,
    )


def set_app_state(state: AppState | None) -> None:
    # FastAPI lifespan 启动时设置，关闭时清空；请求处理只读取这个状态。
    global app_state
    app_state = state


def get_app_state() -> AppState:
    # 路由通过 Depends(get_app_state) 获取启动时初始化好的服务对象。
    if app_state is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG service is not initialized",
        )
    return app_state
