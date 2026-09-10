"""
Recursive, token-aware chunking.

Two ideas carry this module.

*Recursive* means the text is cut on a hierarchy of separators -- paragraphs
first, then lines, then sentences, then words -- and we only step down a level
for the fragments that are still too large. The goal is semantic: chunks end on
a natural boundary instead of mid-sentence, because a chunk that starts with
"...tion have seen an increase" embeds to nothing useful.

*Token-aware* means every size test counts tokens, not characters. Tokens are
what the embedding model actually consumes, and the ratio between the two is not
constant: prose runs about four characters per token, HTML markup far fewer.

Tables are handled separately. The loader renders them as HTML precisely so a
figure stays attached to its fiscal year; splitting one naively would hand the
model `<td>Football costs</td><td>(163,102)</td>` with no year in sight, undoing
the whole point. A table that fits stays whole; a table that does not is cut on
row boundaries with its header re-emitted in every fragment, which costs about
36 tokens -- 7% of a 512-token chunk.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass

import tiktoken

from rag_architect.config import ChunkingSettings
from rag_architect.domain.documents import Chunk, RawDocument

# Strongest to weakest. The empty string is the terminal case: split on raw
# length. Without it, a fragment containing none of the separators would recurse
# forever.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

# The HTML emitted by pymupdf4llm is machine-generated and flat -- no nesting, no
# attributes -- so a regex is enough and a parser would be dead weight.
TABLE_RE = re.compile(r"<table>.*?</table>", re.S)
ROW_RE = re.compile(r"<tr>.*?</tr>", re.S)

# RawDocument.text joins pages with this, and the page index below has to use
# exactly the same separator or every citation drifts by two characters a page.
PAGE_SEPARATOR = "\n\n"

# Page furniture -- a bare page number, a running footer like "- 43 -" -- comes
# out of extraction as its own tiny fragment. It can never answer a question,
# but it can surface on a query containing a number, so it is dropped rather
# than indexed. Three letters is enough to keep any real sentence, and HTML
# table fragments always clear the bar through their own tags.
MIN_ALPHA_CHARACTERS = 3


@dataclass(frozen=True)
class _Piece:
    """
    A chunk-to-be: the text that will be embedded, and the span of the source
    document it came from.

    For prose the two match exactly. For a table fragment they deliberately do
    not: `text` carries a re-emitted header that appears once in the source,
    while `char_start`/`char_end` still point at the rows themselves, so the
    provenance recorded on the chunk stays truthful.
    """

    text: str
    char_start: int
    char_end: int


class _PageIndex:
    """
    Maps a character offset in the joined document text back to a page number.

    Built once per document, then queried once per chunk boundary. `bisect_right`
    over the sorted page-start offsets makes each lookup O(log n) instead of a
    linear scan, which matters because a 157-page filing produces hundreds of
    chunks and each needs two lookups.
    """

    def __init__(self, document: RawDocument) -> None:
        starts: list[int] = []
        offset = 0
        for page in document.pages:
            starts.append(offset)
            offset += len(page.text) + len(PAGE_SEPARATOR)
        self._starts = starts
        self._numbers = [page.number for page in document.pages]

    def page_at(self, offset: int) -> int:
        position = bisect_right(self._starts, offset) - 1
        return self._numbers[max(position, 0)]


class RecursiveTokenChunker:
    """
    Splits a document into embeddable chunks that keep their page provenance.

    Takes a `ChunkingSettings` rather than two loose integers on purpose: that
    model already guarantees `chunk_overlap < chunk_size`, so the invariant is
    carried by the type and this class has no reason to re-check it. Making the
    invalid state unrepresentable beats validating it twice.
    """

    def __init__(self, settings: ChunkingSettings) -> None:
        self._settings = settings
        self._encoding = tiktoken.get_encoding(settings.encoding_name)

    # ---------------------------------------------------------------- public

    def split(self, document: RawDocument) -> list[Chunk]:
        text = document.text
        pages = _PageIndex(document)
        content_hash = document.content_hash
        document_id = document.metadata.document_id

        pieces: list[_Piece] = []
        for start, end, is_table in self._segments(text):
            if is_table:
                pieces.extend(self._split_table(text, start, end))
            else:
                pieces.extend(self._split_prose(text, start, end))

        chunks: list[Chunk] = []
        for piece in pieces:
            if not self._is_indexable(piece.text):
                continue
            page_start = pages.page_at(piece.char_start)
            page_end = pages.page_at(max(piece.char_end - 1, piece.char_start))
            chunks.append(
                Chunk(
                    id=Chunk.build_id(document_id, page_start, piece.text),
                    text=piece.text,
                    metadata=document.metadata,
                    content_hash=content_hash,
                    page_start=page_start,
                    page_end=max(page_end, page_start),
                    char_start=piece.char_start,
                    char_end=piece.char_end,
                )
            )
        return chunks

    @staticmethod
    def _is_indexable(text: str) -> bool:
        """Reject page furniture: fragments carrying no readable words."""
        return sum(1 for character in text if character.isalpha()) >= MIN_ALPHA_CHARACTERS

    # ------------------------------------------------------------ segmenting

    def _segments(self, text: str) -> list[tuple[int, int, bool]]:
        """Alternating spans of prose and whole tables, in document order."""
        spans: list[tuple[int, int, bool]] = []
        cursor = 0
        for match in TABLE_RE.finditer(text):
            if match.start() > cursor:
                spans.append((cursor, match.start(), False))
            spans.append((match.start(), match.end(), True))
            cursor = match.end()
        if cursor < len(text):
            spans.append((cursor, len(text), False))
        return spans

    # ----------------------------------------------------------------- prose

    def _split_prose(self, text: str, start: int, end: int) -> list[_Piece]:
        atoms = self._atomise(text[start:end], SEPARATORS)
        return self._merge_with_overlap(atoms, start)

    def _atomise(self, segment: str, separators: tuple[str, ...]) -> list[str]:
        """
        Cut `segment` into fragments that each fit in chunk_size.

        The fragments concatenate back to `segment` exactly -- separators stay
        attached to the fragment they follow -- which is what lets the caller
        recover character offsets by simple accumulation.
        """
        if self._n_tokens(segment) <= self._settings.chunk_size:
            return [segment]
        if not separators:
            return self._split_by_length(segment)

        separator, rest = separators[0], separators[1:]
        if separator == "":
            return self._split_by_length(segment)

        parts = segment.split(separator)
        parts = [part + separator for part in parts[:-1]] + [parts[-1]]

        fragments: list[str] = []
        for part in parts:
            if not part:
                continue
            if self._n_tokens(part) <= self._settings.chunk_size:
                fragments.append(part)
            else:
                fragments.extend(self._atomise(part, rest))
        return fragments

    def _split_by_length(self, segment: str) -> list[str]:
        """
        Terminal case: no separator left, cut on raw length.

        Only reached by text with no whitespace at all -- a long unbroken
        identifier, or OCR noise. Slicing on characters rather than on decoded
        token ranges keeps the fragments byte-identical to the source, so offsets
        stay exact.
        """
        tokens = self._n_tokens(segment)
        if tokens == 0:
            return [segment]
        # Aim slightly under the limit, then shrink until the estimate holds.
        step = max(1, int(len(segment) * self._settings.chunk_size / tokens * 0.9))
        fragments: list[str] = []
        cursor = 0
        while cursor < len(segment):
            width = step
            while (
                width > 1
                and self._n_tokens(segment[cursor : cursor + width]) > self._settings.chunk_size
            ):
                width //= 2
            fragments.append(segment[cursor : cursor + width])
            cursor += width
        return fragments

    def _merge_with_overlap(self, atoms: list[str], offset: int) -> list[_Piece]:
        """
        Pack fragments into chunks, carrying the tail of each chunk into the next.

        The overlap exists so that a fact straddling a boundary appears whole in
        at least one chunk. It is a proximity window and nothing more: it cannot
        recover context that sits further back than `chunk_overlap` tokens.
        """
        chunk_size = self._settings.chunk_size
        overlap = self._settings.chunk_overlap

        pieces: list[_Piece] = []
        current: list[tuple[int, str]] = []
        current_tokens = 0
        cursor = offset

        for atom in atoms:
            atom_start = cursor
            cursor += len(atom)
            atom_tokens = self._n_tokens(atom)

            if current and current_tokens + atom_tokens > chunk_size:
                pieces.append(self._assemble(current))
                current, current_tokens = self._carry_over(current, overlap)

            current.append((atom_start, atom))
            current_tokens += atom_tokens

        if current:
            pieces.append(self._assemble(current))
        return pieces

    def _carry_over(
        self, current: list[tuple[int, str]], overlap: int
    ) -> tuple[list[tuple[int, str]], int]:
        """Trailing fragments of the finished chunk that seed the next one."""
        carried: list[tuple[int, str]] = []
        carried_tokens = 0
        for entry in reversed(current):
            tokens = self._n_tokens(entry[1])
            if carried_tokens + tokens > overlap:
                break
            carried.insert(0, entry)
            carried_tokens += tokens
        return carried, carried_tokens

    @staticmethod
    def _assemble(entries: list[tuple[int, str]]) -> _Piece:
        text = "".join(entry[1] for entry in entries)
        char_start = entries[0][0]
        return _Piece(text=text, char_start=char_start, char_end=char_start + len(text))

    # ---------------------------------------------------------------- tables

    def _split_table(self, text: str, start: int, end: int) -> list[_Piece]:
        table = text[start:end]
        if self._n_tokens(table) <= self._settings.chunk_size:
            return [_Piece(text=table, char_start=start, char_end=end)]

        rows = list(ROW_RE.finditer(table))
        # A table we cannot read as rows is better handled as prose than dropped.
        if len(rows) < 2:
            return self._split_prose(text, start, end)

        header = rows[0].group(0)
        budget = self._settings.chunk_size - self._n_tokens(f"<table>{header}</table>")

        pieces: list[_Piece] = []
        body: list[str] = []
        body_tokens = 0
        span_start = start + rows[1].start()
        span_end = span_start

        for match in rows[1:]:
            row = match.group(0)
            row_tokens = self._n_tokens(row)
            if body and body_tokens + row_tokens > budget:
                pieces.append(self._assemble_table(header, body, span_start, span_end))
                body, body_tokens = [], 0
                span_start = start + match.start()
            body.append(row)
            body_tokens += row_tokens
            span_end = start + match.end()

        if body:
            pieces.append(self._assemble_table(header, body, span_start, span_end))
        return pieces

    @staticmethod
    def _assemble_table(header: str, rows: list[str], start: int, end: int) -> _Piece:
        """Each fragment is a standalone, valid HTML table carrying the header."""
        return _Piece(
            text=f"<table>{header}{''.join(rows)}</table>",
            char_start=start,
            char_end=end,
        )

    # ---------------------------------------------------------------- tokens

    def _n_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))
