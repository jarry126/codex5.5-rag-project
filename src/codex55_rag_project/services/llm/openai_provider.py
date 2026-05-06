"""OpenAI 兼容类向后兼容别名。

中文：兼容旧导入路径，避免已经引用 openai_provider 的代码在迁移时直接断裂。
English: Backward-compatible aliases keep existing imports working during migration.
"""

from __future__ import annotations

from codex55_rag_project.services.llm.openai_compatible import OpenAICompatibleChatLLM, OpenAICompatibleEmbedder


# 中文：兼容旧导入路径，避免已经引用 openai_provider 的代码在迁移时直接断裂。
# English: Backward-compatible aliases keep existing imports working during migration.
OpenAIEmbedder = OpenAICompatibleEmbedder
OpenAIChatLLM = OpenAICompatibleChatLLM
