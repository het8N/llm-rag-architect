"""
The ports of the hexagon.

Every one of these is a `Protocol`, not an abstract base class: adapters satisfy
them structurally, by having the right methods, without importing or inheriting
from anything here. The dependency arrow points inward only -- the Qdrant
adapter knows about the domain, the domain knows nothing about Qdrant -- and a
test fake is an ordinary class with three methods, not a subclass.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from rag_architect.domain.documents import Chunk, DocumentType, RawDocument
from rag_architect.domain.retrieval import SearchResult


class DocumentLoader(Protocol):
    """Turns a file on disk into pages of text."""

    def load(
        self,
        path: Path,
        *,
        document_type: DocumentType,
        title: str,
        club: str | None = None,
        fiscal_year: int | None = None,
    ) -> RawDocument: ...


class Chunker(Protocol):
    """Splits a document into embeddable chunks that keep their page provenance."""

    def split(self, document: RawDocument) -> list[Chunk]: ...


class Embedder(Protocol):
    """
    Turns text into vectors.

    Documents and queries go through separate methods on purpose: retrieval
    models of the BGE family are trained asymmetrically and expect an
    instruction prefix on the query side only. Embedding a question exactly like
    a passage measurably degrades recall, and the difference is invisible unless
    the port forces the distinction.
    """

    @property
    def dimension(self) -> int:
        """Vector width, read from the loaded model rather than configured."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """Stores vectors with their payload and searches them."""

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if absent. Must be safe to call repeatedly."""
        ...

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """Insert or replace by chunk id, and return how many points were written."""
        ...

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        document_type: DocumentType | None = None,
    ) -> list[SearchResult]: ...


class LLMClient(Protocol):
    """
    A chat completion behind an OpenAI-compatible API.

    Narrow by design: the domain needs one call. Ollama locally and Groq in
    production both satisfy it, which is what makes the switch a matter of two
    environment variables.
    """

    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...
