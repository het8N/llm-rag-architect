"""
Loader tests build their own PDFs.

The real corpus is gitignored (22 MB of filings), so a test that read from
data/raw would pass on Nathan's laptop and fail in CI. Synthesising a two-page
PDF with PyMuPDF keeps the suite hermetic and fast.
"""

from pathlib import Path

import pymupdf
import pytest

from rag_architect.domain.documents import DocumentType
from rag_architect.ingestion.loaders.pdf import PyMuPDFLoader


def write_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        if body:
            page.insert_text((72, 100), body, fontsize=11)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def loader() -> PyMuPDFLoader:
    return PyMuPDFLoader()


def test_load_returns_one_page_per_pdf_page_numbered_from_one(loader, tmp_path):
    path = write_pdf(tmp_path / "accounts.pdf", ["Turnover 221,293", "Football costs"])

    document = loader.load(path, document_type=DocumentType.CLUB_ACCOUNTS, title="Test accounts")

    assert [page.number for page in document.pages] == [1, 2]
    assert "221,293" in document.pages[0].text


def test_metadata_is_carried_through(loader, tmp_path):
    path = write_pdf(tmp_path / "brighton24.pdf", ["Turnover"])

    document = loader.load(
        path,
        document_type=DocumentType.CLUB_ACCOUNTS,
        title="Brighton 2024",
        club="Brighton",
        fiscal_year=2024,
    )

    assert document.metadata.document_id == "brighton24"
    assert document.metadata.club == "Brighton"
    assert document.metadata.fiscal_year == 2024
    assert document.metadata.document_type is DocumentType.CLUB_ACCOUNTS


def test_content_hash_is_stable_across_two_loads(loader, tmp_path):
    """Re-ingesting an untouched file must not look like a changed document."""
    path = write_pdf(tmp_path / "accounts.pdf", ["Turnover 221,293"])

    first = loader.load(path, document_type=DocumentType.CLUB_ACCOUNTS, title="t")
    second = loader.load(path, document_type=DocumentType.CLUB_ACCOUNTS, title="t")

    assert first.content_hash == second.content_hash


def test_a_pdf_without_a_text_layer_is_rejected(loader, tmp_path):
    """A scan that was never OCR'd must fail loudly, not index as an empty document."""
    path = write_pdf(tmp_path / "scan.pdf", ["", ""])

    with pytest.raises(ValueError, match="no extractable text"):
        loader.load(path, document_type=DocumentType.CLUB_ACCOUNTS, title="t")


def test_a_missing_file_is_rejected(loader, tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "absent.pdf", document_type=DocumentType.CLUB_ACCOUNTS, title="t")
