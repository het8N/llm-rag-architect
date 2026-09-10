"""
Core domain objects.

This module imports nothing but the standard library and Pydantic. It knows
nothing about PDFs, Qdrant, tokenizers or HTTP: every one of those lives behind
a port (see `ports.py`) and is supplied by an adapter. That is what lets the
whole pipeline be exercised in memory, with fakes, without Docker.
"""

import hashlib
import re
import unicodedata
from enum import StrEnum
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")

# Fixed namespace for uuid5. Never change it: every chunk id in every existing
# collection is derived from it, and changing it would silently orphan them all.
CHUNK_NAMESPACE = UUID("6f1a4d2e-0b3c-5a7e-9d18-2c4b6e8a0f31")


class DocumentType(StrEnum):
    """
    Primary sources (a club's own filed accounts) versus secondary analysis
    (an industry report). Indexed in the vector store payload so a query can be
    restricted to one or the other: "what did Brighton report" and "what does
    the market say" are different questions.
    """

    CLUB_ACCOUNTS = "club_accounts"
    INDUSTRY_REPORT = "industry_report"


def normalise(text: str) -> str:
    """
    NFKC-normalise text and repair the artefacts PDF extraction leaves behind.

    NFKC folds the typographic variants a PDF is full of -- ligatures (fi),
    non-breaking and narrow spaces, full-width digits -- onto their plain ASCII
    equivalents, so that "office" and "office" hash and embed identically.
    """
    text = unicodedata.normalize("NFKC", text)
    # Words split across a line break by a hyphen: "deprecia-\ntion".
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    # Collapse runs of spaces and tabs, but keep line structure: the HTML tables
    # produced by the loader rely on their own markup, not on layout, yet blank
    # lines still separate paragraphs.
    text = _SPACES.sub(" ", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


class DocumentMetadata(BaseModel):
    """Provenance of a document, carried all the way into the vector payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    document_type: DocumentType
    title: str = Field(min_length=1)
    club: str | None = Field(default=None)
    fiscal_year: int | None = Field(default=None, ge=1900, le=2100)


class Page(BaseModel):
    """One page of extracted text, numbered as a human would cite it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int = Field(ge=1)
    text: str


class RawDocument(BaseModel):
    """A document after extraction, before chunking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: DocumentMetadata
    pages: tuple[Page, ...]

    @property
    def text(self) -> str:
        """Full text, pages joined in order."""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def content_hash(self) -> str:
        """
        sha256 of the extracted text.

        Cheap way to answer "has this document changed since the last run?"
        without re-chunking or re-embedding anything. Ingestion compares it to
        the hash stored in the vector payload and skips the document when equal.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class Chunk(BaseModel):
    """
    An embeddable slice of a document, with enough provenance to cite it.

    `page_start` / `page_end` are what make "[page 47]" possible; a chunk that
    straddles a page break carries both.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    text: str = Field(min_length=1)
    metadata: DocumentMetadata
    content_hash: str = Field(min_length=64, max_length=64)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    @staticmethod
    def build_id(document_id: str, page_start: int, text: str) -> UUID:
        """
        Deterministic id: the same slice of the same document always maps to the
        same UUID, so re-ingesting is an idempotent upsert rather than a
        duplicate insert, and unchanged chunks are never re-embedded.

        The name is deliberately built from the chunk *text* rather than its
        character offsets: inserting a sentence at the top of a document would
        shift every offset and invalidate every id, whereas hashing the text
        leaves untouched chunks untouched. `page_start` disambiguates a header
        or footer repeated verbatim on several pages.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return uuid5(CHUNK_NAMESPACE, f"{document_id}|{page_start}|{digest}")
