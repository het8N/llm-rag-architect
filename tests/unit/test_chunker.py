import re

import pytest
import tiktoken

from rag_architect.config import ChunkingSettings
from rag_architect.domain.documents import DocumentMetadata, DocumentType, Page, RawDocument
from rag_architect.ingestion.chunking.recursive import RecursiveTokenChunker

ENCODING = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def make_document(*page_texts: str) -> RawDocument:
    return RawDocument(
        metadata=DocumentMetadata(
            document_id="brighton24",
            document_type=DocumentType.CLUB_ACCOUNTS,
            title="Brighton 2024",
            club="Brighton",
            fiscal_year=2024,
        ),
        pages=tuple(Page(number=i, text=t) for i, t in enumerate(page_texts, start=1)),
    )


def paragraphs(count: int, words: int = 60) -> str:
    return "\n\n".join(" ".join(f"word{i}{j}" for j in range(words)) for i in range(count))


@pytest.fixture
def chunker() -> RecursiveTokenChunker:
    return RecursiveTokenChunker(ChunkingSettings(chunk_size=128, chunk_overlap=16))


class TestProse:
    def test_short_document_stays_one_chunk(self, chunker):
        chunks = chunker.split(make_document("Turnover was 221,293 in 2024."))

        assert len(chunks) == 1
        assert chunks[0].text.strip() == "Turnover was 221,293 in 2024."

    def test_every_chunk_respects_chunk_size(self, chunker):
        chunks = chunker.split(make_document(paragraphs(12)))

        assert len(chunks) > 1
        assert all(n_tokens(c.text) <= 128 for c in chunks)

    def test_chunks_are_cut_on_paragraph_boundaries(self, chunker):
        """A chunk should not begin in the middle of a word."""
        chunks = chunker.split(make_document(paragraphs(8)))

        assert all(c.text.strip().startswith("word") for c in chunks)

    def test_consecutive_chunks_overlap(self, chunker):
        chunks = chunker.split(make_document(paragraphs(12)))

        # The tail of one chunk reappears at the head of the next.
        assert any(
            chunks[i + 1].text.strip()[:20] in chunks[i].text for i in range(len(chunks) - 1)
        )

    def test_no_overlap_when_configured_to_zero(self):
        chunker = RecursiveTokenChunker(ChunkingSettings(chunk_size=128, chunk_overlap=0))
        chunks = chunker.split(make_document(paragraphs(12)))
        rebuilt = "".join(c.text for c in chunks)

        # With no overlap the chunks tile the document exactly, nothing repeated.
        assert len(rebuilt) == len(chunks[0].text) * 0 + sum(len(c.text) for c in chunks)
        assert all(chunks[i].char_end <= chunks[i + 1].char_start for i in range(len(chunks) - 1))

    def test_text_with_no_separators_is_still_split(self, chunker):
        """Terminal case: a long unbroken string must not loop forever."""
        chunks = chunker.split(make_document("x" * 4000))

        assert len(chunks) > 1
        assert all(n_tokens(c.text) <= 128 for c in chunks)


class TestPages:
    def test_page_provenance_is_exact(self, chunker):
        document = make_document("First page.", "Second page.", "Third page.")
        chunks = chunker.split(document)

        assert [(c.page_start, c.page_end) for c in chunks] == [(1, 3)]

    def test_each_chunk_lands_on_its_own_page(self, chunker):
        document = make_document(paragraphs(6), paragraphs(6))
        chunks = chunker.split(document)

        pages = {c.page_start for c in chunks}
        assert pages == {1, 2}

    def test_page_numbers_follow_the_document_not_the_index(self, chunker):
        """Pages keep the numbers the loader gave them, whatever their order."""
        document = RawDocument(
            metadata=make_document("x").metadata,
            pages=(Page(number=17, text=paragraphs(6)), Page(number=18, text=paragraphs(6))),
        )
        chunks = chunker.split(document)

        assert min(c.page_start for c in chunks) == 17
        assert max(c.page_end for c in chunks) == 18


class TestTables:
    HEADER = "<tr><th></th><th>2024</th><th>2023</th></tr>"

    def table(self, rows: int) -> str:
        body = "".join(
            f"<tr><td>Cost line number {i}</td><td>{i}1,293</td><td>{i}3,574</td></tr>"
            for i in range(rows)
        )
        return f"<table>{self.HEADER}{body}</table>"

    def test_a_small_table_is_never_split(self, chunker):
        chunks = chunker.split(make_document(self.table(3)))

        assert len(chunks) == 1
        assert chunks[0].text.count("<table>") == 1

    def test_a_large_table_is_split_on_row_boundaries(self, chunker):
        chunks = chunker.split(make_document(self.table(40)))

        assert len(chunks) > 1
        for chunk in chunks:
            # No fragment ever cuts through a row.
            assert chunk.text.count("<tr>") == chunk.text.count("</tr>")
            assert not re.search(r"<tr>(?:(?!</tr>).)*$", chunk.text, re.S)

    def test_every_fragment_carries_the_header(self, chunker):
        """This is the whole point: a figure must keep its fiscal year."""
        chunks = chunker.split(make_document(self.table(40)))

        assert len(chunks) > 1
        assert all(self.HEADER in chunk.text for chunk in chunks)
        assert all(chunk.text.startswith("<table>") for chunk in chunks)
        assert all(chunk.text.endswith("</table>") for chunk in chunks)

    def test_no_data_row_is_lost(self, chunker):
        chunks = chunker.split(make_document(self.table(40)))
        emitted = "".join(chunks_text := [c.text for c in chunks])

        for i in range(40):
            assert f"Cost line number {i}</td>" in emitted, i
        assert len(chunks_text) > 1

    def test_prose_around_a_table_is_kept_separate(self, chunker):
        document = make_document(f"Business review\n\n{self.table(3)}\n\nFinancial highlights")
        chunks = chunker.split(document)

        table_chunks = [c for c in chunks if "<table>" in c.text]
        assert len(table_chunks) == 1
        assert "Business review" not in table_chunks[0].text


class TestIdentity:
    def test_chunking_is_deterministic(self, chunker):
        document = make_document(paragraphs(10))

        first = chunker.split(document)
        second = chunker.split(document)

        assert [c.id for c in first] == [c.id for c in second]

    def test_chunk_ids_are_unique_within_a_document(self, chunker):
        chunks = chunker.split(make_document(paragraphs(20)))

        assert len({c.id for c in chunks}) == len(chunks)

    def test_content_hash_is_the_documents(self, chunker):
        document = make_document(paragraphs(6))
        chunks = chunker.split(document)

        assert all(c.content_hash == document.content_hash for c in chunks)


class TestPageFurniture:
    """Bare page numbers and running footers must never reach the index."""

    @pytest.mark.parametrize("furniture", ["26", "- 43 -", "  \n 12 \n ", "..."])
    def test_furniture_is_dropped(self, chunker, furniture):
        chunks = chunker.split(make_document(f"A real sentence with words.\n\n{furniture}"))

        assert all(furniture.strip() != c.text.strip() for c in chunks)

    def test_short_but_real_sentences_survive(self, chunker):
        chunks = chunker.split(make_document("Net debt rose."))

        assert len(chunks) == 1
        assert chunks[0].text.strip() == "Net debt rose."

    def test_a_numeric_table_fragment_is_kept(self, chunker):
        """HTML tags carry letters, so a table of pure figures still qualifies."""
        table = "<table><tr><th>2024</th></tr><tr><td>221,293</td></tr></table>"
        chunks = chunker.split(make_document(table))

        assert len(chunks) == 1
        assert "221,293" in chunks[0].text
