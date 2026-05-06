"""文档加载器。

中文：从文件系统或内存加载文档，实现 ports.DocumentLoader 协议。
English: Loads documents from filesystem or memory, implementing ports.DocumentLoader protocol.
"""

from __future__ import annotations

from pathlib import Path

from codex55_rag_project.core.models import Document


class TextDirectoryLoader:
    """从目录加载文本文件。

    中文：扫描指定目录下的文本文件，读取为 Document 对象列表。
    English: Scans directory for text files, reads them as Document list.
    """

    def __init__(self, directory: str | Path, glob: str = "*.txt") -> None:
        self.directory = Path(directory)
        self.glob = glob

    def load(self) -> list[Document]:
        """加载目录下所有匹配的文本文件。

        中文：按路径排序读取，metadata 包含 source 和 filename。
        English: Reads matching files sorted by path; metadata includes source and filename.
        """
        documents: list[Document] = []
        for path in sorted(self.directory.glob(self.glob)):
            documents.append(
                Document(
                    id=str(path),
                    text=path.read_text(encoding="utf-8"),
                    metadata={"source": str(path), "filename": path.name},
                )
            )
        return documents


class StaticDocumentLoader:
    """从内存加载预置文档。

    中文：用于测试或 API 请求场景，直接传入已构造好的 Document 列表。
    English: For testing or API requests, directly passes pre-constructed Document list.
    """

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def load(self) -> list[Document]:
        """返回构造时传入的文档列表。"""
        return self.documents

