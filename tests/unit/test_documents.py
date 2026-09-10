import pytest
from pydantic import ValidationError

from rag_architect.domain.documents import (
    Chunk,
    DocumentMetadata,
    DocumentType,
    Page,
    RawDocument,
    normalise,
)


def make_metadata(document_id: str = "brighton24") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        document_type=DocumentType.CLUB_ACCOUNTS,
        title="Brighton & Hove Albion FC Limited",
        club="Brighton",
        fiscal_year=2024,
    )


class TestNormalise:
    def test_nfkc_folds_typographic_variants(self):
        """A non-breaking space and a ligature must hash like their plain forms."""
        assert normalise("221\u00a0293") == "221 293"
        assert normalise("of\ufb01ce") == "office"

    def test_hyphenation_across_a_line_break_is_repaired(self):
        assert normalise("deprecia-\ntion") == "depreciation"

    def test_runs_of_spaces_collapse_but_paragraphs_survive(self):
        assert normalise("a    b") == "a b"
        assert normalise("para one\n\n\n\npara two") == "para one\n\npara two"

    def test_currency_and_digits_are_left_alone(self):
        """Normalisation must never touch the figures: they are the payload."""
        assert normalise("Turnover £221,293 (42,700)") == "Turnover £221,293 (42,700)"


class TestRawDocument:
    def test_content_hash_depends_only_on_text(self):
        pages = (Page(number=1, text="hello"), Page(number=2, text="world"))
        first = RawDocument(metadata=make_metadata(), pages=pages)
        # Same text, different provenance: re-titling a file must not force a
        # full re-ingestion.
        second = RawDocument(metadata=make_metadata("other-id"), pages=pages)

        assert first.content_hash == second.content_hash
        assert len(first.content_hash) == 64

    def test_content_hash_changes_when_text_changes(self):
        base = RawDocument(metadata=make_metadata(), pages=(Page(number=1, text="hello"),))
        edited = RawDocument(metadata=make_metadata(), pages=(Page(number=1, text="hellp"),))

        assert base.content_hash != edited.content_hash

    def test_documents_are_immutable(self):
        document = RawDocument(metadata=make_metadata(), pages=(Page(number=1, text="a"),))

        with pytest.raises(ValidationError):
            document.pages = ()

    def test_page_numbering_starts_at_one(self):
        with pytest.raises(ValidationError):
            Page(number=0, text="a")


class TestChunkId:
    def test_same_slice_yields_the_same_id(self):
        """This is what makes re-ingestion an idempotent upsert."""
        assert Chunk.build_id("brighton24", 3, "Turnover 221,293") == Chunk.build_id(
            "brighton24", 3, "Turnover 221,293"
        )

    def test_id_changes_with_text_document_or_page(self):
        base = Chunk.build_id("brighton24", 3, "Turnover 221,293")

        assert base != Chunk.build_id("brighton24", 3, "Turnover 203,574")
        assert base != Chunk.build_id("chelsea25", 3, "Turnover 221,293")
        assert base != Chunk.build_id("brighton24", 4, "Turnover 221,293")

    def test_id_is_stable_across_processes(self):
        """
        uuid5 is a pure function of namespace and name, unlike uuid4. Hard-coded
        so that a change to CHUNK_NAMESPACE fails the suite loudly rather than
        silently orphaning every chunk already in the collection.
        """
        assert str(Chunk.build_id("brighton24", 3, "Turnover 221,293")) == (
            "ed02b2e9-8998-5811-9234-c3c048853ec1"
        )

    def test_chunk_rejects_a_malformed_content_hash(self):
        with pytest.raises(ValidationError):
            Chunk(
                id=Chunk.build_id("brighton24", 3, "x"),
                text="x",
                metadata=make_metadata(),
                content_hash="too-short",
                page_start=3,
                page_end=3,
                char_start=0,
                char_end=1,
            )
