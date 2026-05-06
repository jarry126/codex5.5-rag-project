"""PDF 文档加载器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from codex55_rag_project.core.models import Document


PdfReaderFactory = Callable[[str | Path], Any]


class PdfDocumentLoader:
    """按页读取 PDF，并生成带来源元数据的 Document。"""

    def __init__(
        self,
        pdf_path: str | Path,
        tenant_id: str,
        ingestion_id: str,
        reader_factory: PdfReaderFactory | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.tenant_id = tenant_id
        self.ingestion_id = ingestion_id
        self.reader_factory = reader_factory

    def load(self) -> list[Document]:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {self.pdf_path}")
        reader = self._build_reader()
        documents: list[Document] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    id=f"{self.pdf_path.stem}-page-{page_index}",
                    text=text,
                    metadata={
                        "source": str(self.pdf_path),
                        "file_name": self.pdf_path.name,
                        "page_number": page_index,
                        "tenant_id": self.tenant_id,
                        "ingestion_id": self.ingestion_id,
                    },
                )
            )
        if not documents:
            raise ValueError(
                "PDF did not contain extractable text; OCR is required before ingestion"
            )
        return documents

    def _build_reader(self) -> Any:
        if self.reader_factory is not None:
            return self.reader_factory(self.pdf_path)
        from pypdf import PdfReader

        return PdfReader(str(self.pdf_path))
