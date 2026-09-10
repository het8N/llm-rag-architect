# llm-rag-architect

Production-minded RAG over the annual financial filings of five Premier League
clubs. Ask a question about turnover, player trading or broadcasting revenue and
get an answer grounded in the filings, with the page it came from — or an
explicit refusal when the corpus does not support an answer.

No LangChain, no LlamaIndex. Every layer is written directly against the library
underneath it, so every decision in this repository is one I can defend.

> **Status: in progress.** Ingestion works end to end — extraction, chunking and
> the domain model are built and tested. Retrieval, the HTTP API, the container
> and the deployment are not yet written. This section is updated as milestones
> land; the roadmap at the bottom says exactly where the line currently sits.

---

## Why this corpus

Football club accounts are an unusually good stress test for a RAG system. The
answers live in tables — a turnover figure means nothing without its fiscal year
column — and the documents themselves are hostile: four of the six filings in
this corpus are scans with no text layer at all.

The subject is worth reading too: private equity entering English football, the
weight of broadcasting rights, and how player trading is accounted for.

## Corpus

| Document | Pages | Source | Text layer |
|---|---|---|---|
| Brighton & Hove Albion FC, FY2024 | 35 | Companies House | OCR'd |
| Brighton & Hove Albion FC, FY2025 | 37 | Companies House | OCR'd |
| Chelsea FC, FY2025 | 46 | Companies House | OCR'd |
| Newcastle United FC, FY2025 | 47 | Companies House | OCR'd |
| Manchester United plc, FY2025 (Form 20-F) | 157 | SEC EDGAR | native |
| Deloitte Annual Review of Football Finance 2026 | 56 | Deloitte | native |

The PDFs are not committed — they are 22 MB and freely available from the
sources above. Filings retrieved from Companies House arrive as scanned TIFFs
wrapped in a PDF (`producer: libtiff / tiff2pdf`, zero extractable characters)
and were OCR'd once, ahead of ingestion:

```bash
ocrmypdf --language eng --deskew input.pdf output-ocr.pdf
```

OCR accuracy was checked arithmetically rather than by eye: on Brighton's income
statement, both fiscal-year columns still add up after OCR
(`221,293 + 2,450 − 42,700 = 181,043`, and so on down the table). Digits survived
intact.

## Architecture

Hexagonal. `domain/` imports nothing but the standard library and Pydantic; every
external system sits behind a `Protocol` in `domain/ports.py` and is supplied by
an adapter. The dependency arrow points inward only, so the whole pipeline can be
driven with in-memory fakes — no Docker, no network, no model download.

```
                  ┌─────────────────────────────────┐
                  │  domain/                        │
   adapters  ───▶ │    documents.py  retrieval.py   │
   satisfy        │    ports.py  (Protocols)        │
   the ports      └─────────────────────────────────┘
                                 ▲
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   PyMuPDFLoader        RecursiveTokenChunker      (fastembed, Qdrant,
   ingestion/loaders    ingestion/chunking          LLM client — pending)
```

Pipeline, left to right:

```
PDF ──▶ pages ──▶ chunks ──▶ vectors ──▶ Qdrant ──▶ top-k ──▶ answer
        HTML       512 tok    384 dim     ANN        k=5      + [page N]
        tables
```

## Decisions

Recorded as ADRs in [`docs/adr/`](docs/adr/), with the measurements behind them.

**Tables are extracted as HTML.** Plain text flattens a financial table to one
value per line, so `Turnover / 221,293 / 203,574` no longer says which figure is
2024. Markdown pipe tables keep the columns but strip the spaces inside labels
(`Administrativeandoperationalcosts`). Only `table_output="html"` preserved both,
at a cost of 10–23% more tokens and roughly two minutes of one-off ingestion for
the whole corpus. Full comparison in
[ADR 001](docs/adr/001-pdf-extraction-strategy.md).

**A table that outgrows a chunk keeps its header.** A quarter of Brighton's
tables exceed 512 tokens. Cut naively, the second fragment reads
`<td>Football costs</td><td>(163,102)</td>` with no year in sight — reintroducing
exactly the ambiguity HTML extraction removed. Oversized tables are cut on `</tr>`
boundaries with the header re-emitted in every fragment, which costs about 36
tokens, 7% of a chunk. Overlap cannot substitute: on a 730-token table, cutting at
512 leaves the header 448 tokens behind, and overlap reaches back 64. It is a
proximity window, and a table header is a long-distance dependency.

**Chunk ids are deterministic.** `uuid5` over a fixed namespace, keyed on document
id, page and chunk text — not on character offsets, which would all shift when a
sentence is inserted at the top of a file. Re-ingesting is therefore an idempotent
upsert, and unchanged chunks are never re-embedded.

**Qdrant over Chroma.** A real server with payload filtering and quantization,
rather than an embedded library that would have to be replaced before deployment.

**fastembed over sentence-transformers.** ONNX inference needs no PyTorch: a
~400 MB image instead of ~6 GB. Nothing here trains a model.

**One OpenAI-compatible client.** Ollama (`qwen2.5:7b`) locally, Groq in
production. Switching is two environment variables and no code change. Ollama in
production was ruled out: CPU-only serverless inference puts p95 near 90 s.

**Settings are composed, not flat.** Five frozen sub-models behind one
`BaseSettings`, so each component receives only what it needs — the chunker is
handed a `ChunkingSettings` and can never reach the LLM API key.

### Deliberately out of scope

**Kubernetes.** One stateless container behind Cloud Run, scaling to zero. K8s
would add a control plane, an ingress and a cost floor to a service with no
availability requirement.

**Fine-tuning.** RAG changes what a model *knows*; fine-tuning changes what it
*does*. The problem here is knowledge the model does not have, and answers that
must be traceable to a page — the case fine-tuning is worst at.

**LLM-as-judge / RAGAS.** Retrieval is measured instead: `recall@k` and MRR
against a hand-annotated golden dataset. Deterministic, free, and it does not
evaluate one language model with another.

**LangChain / LlamaIndex.** The abstractions they provide are the ones this
project exists to understand.

## Getting started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # create .venv and install, from the committed lockfile
cp .env.example .env     # defaults give a working local setup
uv run pytest            # 47 tests
uv run ruff check .
uv run mypy
```

Configuration is read from the environment, prefixed and nested:
`RAG_CHUNKING__CHUNK_SIZE`, `RAG_LLM__BASE_URL`. Every variable is documented with
its default in [`.env.example`](.env.example).

## Quality

| | |
|---|---|
| Tests | 47 passing, 89% coverage |
| Typing | `mypy --strict`, no ignored modules beyond untyped third-party calls |
| Linting | `ruff` with `E,F,I,UP,B,SIM,ANN,RUF` |
| Dependencies | pinned in `uv.lock`, committed for reproducible builds |

Loader tests synthesise their own PDFs rather than reading `data/raw`, which is
gitignored: a test that read the real corpus would pass locally and fail in CI.

## Roadmap

**Done**

- [x] Composed, validated, frozen settings (`config.py`)
- [x] Domain model and hexagonal ports
- [x] PDF loader — HTML tables, page-level provenance, OCR'd scans rejected if empty
- [x] Recursive token-aware chunker with header-preserving table splitting
- [x] ADR 001 — extraction strategy

**Next**

- [ ] Qdrant via Docker Compose
- [ ] fastembed adapter (`bge-small-en-v1.5`, 384 dimensions)
- [ ] Qdrant adapter — collection, indexed payload, idempotent upsert
- [ ] Ingestion pipeline and `rag ingest`
- [ ] LLM client, grounding prompt, citations and refusal
- [ ] `rag ask` — the first end-to-end answer
- [ ] FastAPI: `/health`, `/query`, `/sources`, API key in header
- [ ] Multi-stage Dockerfile, non-root, healthcheck
- [ ] GitHub Actions, build and push to Artifact Registry
- [ ] Cloud Run deployment with Secret Manager
- [ ] Golden dataset — 30 hand-annotated questions with expected pages
- [ ] `rag eval` — recall@k, MRR, and a CI gate on recall@5
- [ ] Latency p50/p95/p99, broken down by stage, and cost per query
- [ ] Ablation: chunk_size 256/512/1024 × overlap 0/64

**Phase 2, argued rather than attempted**

Hybrid BM25 retrieval, a cross-encoder reranker, table-aware parsing with
Docling, per-chunk incremental reindexing, SSE streaming, Langfuse tracing, query
rewriting, and an embedding cache. Each is deferred because the evaluation
harness does not yet exist to prove it helps.

## Licence

MIT for the code. The filings belong to their respective publishers and are not
redistributed here.
