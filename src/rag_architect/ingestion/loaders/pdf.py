"""PDF loader built on pymupdf4llm. See docs/adr/001-pdf-extraction-strategy.md."""

from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm

from rag_architect.domain.documents import (
    DocumentMetadata,
    DocumentType,
    Page,
    RawDocument,
    normalise,
)


class PyMuPDFLoader:
    """
    Extracts one text block per page, with financial tables rendered as HTML.

    `table_output="html"` is the only variant measured that keeps both the
    column structure and the cell labels intact: plain text flattens a table to
    one value per line, losing which fiscal year a figure belongs to, and
    markdown pipe tables strip the spaces inside labels
    ("Administrativeandoperationalcosts").

    `use_ocr=False` stops pymupdf4llm running its own Tesseract pass. Scanned
    filings are OCR'd once, ahead of time, with OCRmyPDF; keeping OCR out of the
    runtime keeps Tesseract out of the container image.
    """

    def __init__(self, *, table_output: str = "html", use_ocr: bool = False) -> None:
        self._table_output = table_output
        self._use_ocr = use_ocr

    def load(
        self,
        path: Path,
        *,
        document_type: DocumentType,
        title: str,
        club: str | None = None,
        fiscal_year: int | None = None,
    ) -> RawDocument:
        if not path.is_file():
            msg = f"no such PDF: {path}"
            raise FileNotFoundError(msg)

        # PyMuPDF ships no type information; the call is untyped under mypy strict.
        with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
            page_chunks: list[dict[str, Any]] = pymupdf4llm.to_markdown(
                doc,
                page_chunks=True,
                table_output=self._table_output,
                use_ocr=self._use_ocr,
                show_progress=False,
            )

        pages = tuple(
            Page(number=int(chunk["metadata"]["page_number"]), text=normalise(chunk["text"]))
            for chunk in page_chunks
        )
        # A scanned PDF that was never OCR'd yields pages with no text at all.
        # Failing here is deliberate: silently indexing an empty document would
        # surface much later as a document that can never be retrieved.
        if not any(page.text for page in pages):
            msg = f"{path} has no extractable text -- is it a scan that still needs OCR?"
            raise ValueError(msg)

        return RawDocument(
            metadata=DocumentMetadata(
                document_id=path.stem,
                document_type=document_type,
                title=title,
                club=club,
                fiscal_year=fiscal_year,
            ),
            pages=pages,
        )
