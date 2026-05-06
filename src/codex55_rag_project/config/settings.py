"""应用配置模块。

中文：使用 pydantic-settings 从环境变量和 .env 文件读取配置，支持多种 API Key 别名兼容。
English: Uses pydantic-settings to read config from env vars and .env, with multiple API key aliases for compatibility.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"

"""
BaseSettings的作用：
     1、默认会从环境变量读取配置，同时也支持.env 文件、系统环境变量、甚至自定义来源。  
     2、将外部环境的大写映射成小写。  （Pydantic 会做转换）    
     3、BaseSettings是把“字段名 → 转换成环境变量名去匹配”                                                                                  
"""


class Settings(BaseSettings):
    """RAG 服务配置。

    中文：所有配置项以 RAG_ 为前缀从环境变量读取，同时支持 DashScope/OpenAI 别名。
    English: All settings read from env vars with RAG_ prefix, supporting DashScope/OpenAI aliases.
    """

    # SettingsConfigDict设置读取的全局规则；Field代表特殊规则，不受全局规则限制

    # SettingsConfigDict 配置读取规格；真正读取数据的是 BaseSettings。
    # env_file 使用项目根目录的绝对路径，避免 PyCharm 从 src/codex55_rag_project 直接运行时读不到 .env。
    model_config = SettingsConfigDict(  # model_config是Pydantic v2 约定的配置变量名
        env_file=ENV_FILE,
        env_prefix="RAG_",
        extra="ignore",
        populate_by_name=True,  # extra:.env环境中有，下面代码中没有，也不会报错。
    ) # populate_by_name代表字段名和别名都可以用来赋值，所以下面可以使用Filed中创建别名，也可以使用environment直接赋值。

    # 基础配置
    environment: str = "local"
    service_name: str = "codex5.5-rag-project"
    log_level: str = "INFO"

    # 中文：生产默认接阿里云百炼；同时保留 OPENAI_* 别名，方便兼容 OpenAI SDK/LangChain 习惯。
    # English: Production defaults to Alibaba Cloud Model Studio while keeping OPENAI_* aliases for SDK compatibility.
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "DASHSCOPE_API_KEY",
            "RAG_DASHSCOPE_API_KEY",
            "OPENAI_API_KEY",
            "RAG_OPENAI_API_KEY",
            "OPENAI_COMPATIBLE_API_KEY",
        ),
    )
    openai_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices(
            "DASHSCOPE_BASE_URL",
            "RAG_DASHSCOPE_BASE_URL",
            "OPENAI_BASE_URL",
            "RAG_OPENAI_BASE_URL",
            "OPENAI_COMPATIBLE_BASE_URL",
        ),
    )
    # Embedding 模型配置
    # 中文：embedding_model 指定向量模型名称，embedding_dimensions 指定向量维度。
    # English: embedding_model is the embedding model name; embedding_dimensions is the vector dimension.
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024

    # Chat 模型配置
    # 中文：chat_model 是生成回答的模型，chat_temperature 控制创造性（0 最严谨）。
    # English: chat_model is the generation model; chat_temperature controls creativity (0 is most deterministic).
    chat_model: str = "qwen-plus"
    chat_temperature: float = 0.0
    request_timeout_seconds: float = 30.0

    # Rerank 模型配置
    # 中文：rerank_model 是重排序模型，rerank_base_url 是独立的 DashScope rerank endpoint。
    # English: rerank_model is the reranker model; rerank_base_url is DashScope's separate rerank endpoint.
    rerank_model: str = "qwen3-rerank"
    rerank_base_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1"
    rerank_enabled: bool = True

    # PostgreSQL/pgvector 配置
    # 中文：postgres_dsn 是数据库连接串，vector_collection_name/table 是向量存储表名。
    # English: postgres_dsn is the DB connection string; vector_collection_name/table is the vector storage table.
    postgres_dsn: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    medical_postgres_dsn: str = "postgresql+psycopg://liushanshan:postgres@localhost:5432/rag_medical"
    medical_admin_postgres_dsn: str = "postgresql://liushanshan:postgres@localhost:5432/postgres"
    medical_database_name: str = "rag_medical"
    vector_collection_name: str = "rag_chunks"
    vector_table: str = "rag_chunks"
    vector_dimensions: int = 1024
    medical_vector_table: str = "medical_rag_chunks"
    medical_audit_table: str = "medical_rag_audit"
    medical_default_pdf_path: str = "中医临床诊疗智能助手.pdf"
    medical_query_rewrite_count: int = 3
    medical_text_search_config: str = "jiebacfg"

    # 切分和检索配置
    # 中文：chunk_size/overlap 控制切分，candidate_k/final_k/min_score 控制检索召回和过滤。
    # English: chunk_size/overlap control chunking; candidate_k/final_k/min_score control retrieval and filtering.
    chunk_size: int = 900
    chunk_overlap: int = 150
    candidate_k: int = 24
    final_k: int = 6
    min_score: float = 0.0

    # API 认证配置
    # 中文：api_key 用于验证请求身份；空值表示不启用认证（仅适合本地开发）。
    # English: api_key authenticates requests; empty value disables auth (only for local dev).
    api_key: str = ""


@lru_cache(maxsize=1)  # lru_cache缓存，保证配置只读取一次，避免重复解析环境变量。
def get_settings() -> Settings:
    """获取配置单例。

    中文：使用 lru_cache 保证配置只读取一次，避免重复解析环境变量。
    """
    return Settings()
