"""文本切分服务。

中文：将长文档切成适合向量检索的片段，chunk 大小和 overlap 会直接影响召回质量。
English: Split long documents into retrieval-sized chunks; size and overlap affect retrieval quality.
"""

from __future__ import annotations

import hashlib

from codex55_rag_project.core.models import Chunk, Document


class RecursiveTextChunker:
    """递归文本切分器。

    中文：按固定大小切分，优先在句子边界（换行、句号、分号）处断开，保留 overlap 防止上下文丢失。
    English: Fixed-size chunking with preference for sentence boundaries, preserving overlap for context continuity.
    """

    def __init__(self, chunk_size: int = 700, overlap: int = 120) -> None:
        # chunk_size 控制每个切片的最大字符数；overlap 控制相邻切片的重叠字符数。
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        """将文档列表切分成 Chunk 列表。

        中文：每个 Chunk 的 metadata 会继承 Document 的 metadata，并追加 chunk_index。
        English: Each Chunk inherits Document metadata and adds chunk_index.
        """
        chunks: list[Chunk] = []
        for document in documents:
            parts = self._split_text(document.text)
            for index, text in enumerate(parts):
                # chunk_id = document_id#index-sha1[:16]，保证全局唯一且可溯源。
                chunk_id = self._chunk_id(document.id, index, text)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        text=text,
                        metadata={**document.metadata, "chunk_index": index},
                    )
                )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """切分单个文本。

        中文：先清理空行，再按 chunk_size 切分，优先在句子边界处断开。
        English: Clean empty lines, then split by chunk_size with preference for sentence boundaries.
        """
        # 清理空白行，保留有内容的行。
        normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if len(normalized) <= self.chunk_size:
            return [normalized] if normalized else []

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            window = normalized[start:end]
            # 优先在换行、句号、分号处切分，避免截断句子中间。
            split_at = max(window.rfind("\n"), window.rfind("。"), window.rfind(". "), window.rfind("; "))
            if split_at > self.chunk_size * 0.5 and end < len(normalized):
                end = start + split_at + 1
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # overlap 确保相邻切片共享部分内容，减少边界信息丢失。
            start = max(end - self.overlap, end) if self.overlap == 0 else max(0, end - self.overlap)
            if start >= end and end < len(normalized):
                start = end
            if end == len(normalized):
                break
        return chunks

    @staticmethod
    def _chunk_id(document_id: str, index: int, text: str) -> str:
        """生成唯一 chunk_id。

        中文：使用 document_id + index + text 内容的 SHA1 前 16 位，防止重复索引同一内容。
        English: SHA1[:16] of document_id:index:text ensures uniqueness and prevents duplicate indexing.
        """
        digest = hashlib.sha1(f"{document_id}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
        return f"{document_id}#{index}-{digest}"

