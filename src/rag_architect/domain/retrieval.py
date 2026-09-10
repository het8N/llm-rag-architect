"""Value objects produced by retrieval and answering."""

from pydantic import BaseModel, ConfigDict, Field

from rag_architect.domain.documents import Chunk


class SearchResult(BaseModel):
    """A chunk returned by the vector store, with its similarity score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk
    score: float


class Citation(BaseModel):
    """A source the answer points at, in a form a reader can verify."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_title: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class Answer(BaseModel):
    """
    The single product of this system: a grounded answer, or an explicit refusal.

    `refused=True` with an empty `citations` is a first-class outcome, not an
    error. A RAG that cannot find support for a question must say so; inventing
    a plausible figure from a financial report is the worst failure mode here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    citations: tuple[Citation, ...] = ()
    refused: bool = False
