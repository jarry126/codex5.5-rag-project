from __future__ import annotations

from pathlib import Path

import pytest

from codex55_rag_project.loaders.pdf import PdfDocumentLoader


class FakePage:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class FakeReader:
    pages = [FakePage("第一页内容"), FakePage(""), FakePage("第二页内容")]


def test_pdf_loader_extracts_pages_with_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "中医临床诊疗智能助手.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    loader = PdfDocumentLoader(
        pdf_path,
        tenant_id="tenant-a",
        ingestion_id="ingestion-1",
        reader_factory=lambda _: FakeReader(),
    )

    documents = loader.load()

    assert [document.id for document in documents] == [
        "中医临床诊疗智能助手-page-1",
        "中医临床诊疗智能助手-page-3",
    ]
    assert documents[0].metadata["tenant_id"] == "tenant-a"
    assert documents[0].metadata["ingestion_id"] == "ingestion-1"
    assert documents[0].metadata["page_number"] == 1
    assert documents[0].metadata["file_name"] == "中医临床诊疗智能助手.pdf"


def test_pdf_loader_reports_ocr_required_when_no_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    class EmptyReader:
        pages = [FakePage("")]

    loader = PdfDocumentLoader(pdf_path, tenant_id="default", ingestion_id="i", reader_factory=lambda _: EmptyReader())

    with pytest.raises(ValueError, match="OCR is required"):
        loader.load()
