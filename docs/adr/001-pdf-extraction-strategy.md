# ADR 001 — PDF extraction strategy

**Status:** accepted — 2026-09-09

## Context

The corpus is 378 pages of Premier League club annual accounts plus one
industry report. Four of the six source documents were pure scans filed at
Companies House (producer `libtiff / tiff2pdf`, zero text layer) and were run
through OCRmyPDF 17.11 / Tesseract 5.5.3 before ingestion. Two are native
(Manchester United's SEC 20-F, the Deloitte review).

Most questions this RAG must answer are about figures held in financial tables,
where each row carries one value per fiscal year. Extraction has to preserve
which column a number belongs to, and has to keep page granularity so answers
can cite `[page N]`.

## Options measured

Measured on `brighton24-ocr.pdf` (35 pages, OCR) and `manchester25.pdf`
(157 pages, native), tokens counted with `cl100k_base`.

| Method | Brighton | Manchester |
|---|---|---|
| `page.get_text("text")` | 0.04 s — 16,132 tok | 0.32 s — 158,807 tok |
| `to_markdown()` | 13.46 s — 16,325 tok (+1%) | 28.97 s — 160,599 tok (+1%) |
| `to_markdown(table_output="html")` | 15.89 s — 19,925 tok (+23%) | 44.76 s — 174,756 tok (+10%) |

Qualitatively, on the Brighton income-statement page:

- **Raw text** flattens the table to one value per line. `Turnover / 221,293 /
  203,574` gives no way to tell which figure is 2024 and which is 2023 once the
  header lands in a different chunk. Also carries OCR noise (`eeee#e.e`).
- **Markdown tables** restore the columns but strip spaces inside cell labels:
  `Otheroperatingincome`, `Administrativeandoperationalcosts`. Unusable — the
  embedding model tokenises those poorly and keyword search can never match them.
- **HTML tables** restore the columns and keep the labels intact:
  `<td>Other operating income</td><td>2,450</td><td>24,878</td>`.

Digits are identical across all three methods; no figure was lost or altered.
OCR accuracy was verified arithmetically: both columns of the Brighton income
statement add up (221,293 + 2,450 − 42,700 = 181,043, and so on down the table).

## Decision

Extract with `pymupdf4llm.to_markdown(doc, page_chunks=True,
table_output="html", use_ocr=False)`.

- `page_chunks=True` returns one dict per page carrying
  `metadata["page_number"]` (1-based), which is what page-level citations and
  the `_PageIndex` lookup are built on.
- `table_output="html"` is the only variant that preserves both column
  structure and cell labels.
- `use_ocr=False` skips pymupdf4llm's own Tesseract pass (~17% faster on the
  OCR'd files) and keeps Tesseract out of the container image. OCR is a
  deliberate, documented preprocessing step, not a runtime dependency.

## Consequences

**Accepted costs.** Extraction is 300–400x slower than raw text, which is
irrelevant: it runs once at ingestion, offline, and totals about two minutes for
the whole corpus. Query-time latency is untouched. Token count grows 10–23%,
which is the real price: HTML tags consume room inside 512-token chunks, so a
long table fills a chunk sooner and is more likely to be split. This is the main
weakness of the decision and the first thing to revisit if retrieval on table
questions underperforms in the evaluation.

**Dependency.** `pymupdf4llm` 1.28 routes through a GNN layout model and pulls
`onnxruntime` (76 MB), `numpy` and `networkx`. `onnxruntime` and `numpy` are
shared with fastembed, so the marginal image cost is `networkx` plus
`pymupdf-layout`. This was accepted here where a 6 GB torch dependency was not:
ONNX inference is two orders of magnitude smaller.

**Not chosen.** A hybrid strategy — raw text by default, HTML only on pages
where a table is detected — would give a better token/quality trade-off. It was
rejected on schedule grounds: two code paths, a detection heuristic to write and
test, for a corpus small enough that the token overhead does not bind. Worth
revisiting if the corpus grows.
