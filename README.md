# OmniCorpus

Clinical RAG over multiple sources — semantic retrieval + **grounded** generation
over a heterogeneous document corpus, with a Streamlit UI that shows the answer
**beside its verbatim sources**.

## Why verbatim sources, always (the point of this app)

In healthcare, an answer alone is not enough. A reader has to be able to check it.

This app never shows a generated answer by itself. Every answer is displayed
**next to the exact, word-for-word source text it came from** — the same text that
was stored in the database and retrieved for the question. Nothing is paraphrased,
summarized, or re-written between storage and display.

In plain terms:

- The model may **only** use the retrieved source passages to make specific
  clinical claims. It is told, in the prompt, never to use clinical facts from its
  own training. If the sources don't support an answer, it says so instead of
  guessing.
- The passages you read in the "Sources" panel are **byte-for-byte identical** to
  what's in the database. What you see is the ground truth, not a retelling of it.
- The answer can be worded differently each time (language models vary), but the
  **cited source text is fixed and authoritative**. Trust rests on the sources,
  not on the prose.

This is the core difference from an ordinary RAG app: the retrieved passage is both
the thing the answer is built from **and** the thing you're shown to verify it. One
field, no gap between "what grounded the answer" and "what you can audit." That is
how clinical responses should be presented — answer and grounding, side by side.

## Stack

Python 3.12 (`uv`) · Haystack 2.x · Docling (multi-format ingest) · local
sentence-transformers embeddings · Postgres + pgvector · local OpenAI-compatible
LLM · Streamlit. See `./spec.md` for the full specification.

## Quick start

### 1. Install
```bash
uv sync
```

### 2. Bring up services
- **Postgres + pgvector** — either the bundled container or your own server:
  ```bash
  docker compose up -d        # ./docker-compose.yml
  ```
- **A local OpenAI-compatible LLM** (llama.cpp / vLLM / Ollama) serving a `/v1`
  endpoint.

### 3. Configure
Copy the template and fill in your services:
```bash
cp .env.example .env
```
Key settings (`./.env`):
```
PG_CONN_STR=postgresql://USER:PASSWORD@HOST:5432/DB
EMBED_MODEL_ID=BAAI/bge-base-en-v1.5
LLM_BASE_URL=http://HOST:8000/v1
LLM_MODEL=your-model-name
CORPUS_SOURCES=[{"adapter": "local_path", "root": "tests/data/**/*"}]
```

### 4. Ingest a corpus
Put your documents where `CORPUS_SOURCES` points (PDF, DOCX, HTML, …), then:
```bash
uv run corpus-rag ingest          # add --reset to recreate the table
```

A tiny **synthetic sample** ships in the repo at
`./tests/data/sample-clinical-guideline.pdf` so a fresh clone always has
something to ingest, chunk, and process — no need to supply your own corpus to
see the pipeline work end to end. It is generated, non-PHI, common-knowledge
reference text (two headed sections); regenerate it with
`uv run python scripts/make_sample_pdf.py`. Real or licensed corpora are
gitignored — drop them in `./tests/data/` and they'll be picked up too.

With the default `.env` (below) `CORPUS_SOURCES` already points at
`tests/data/**/*`, so out of the box `uv run corpus-rag ingest` indexes that
sample. To make `.env` work you only need: a reachable `PG_CONN_STR`
(Postgres + pgvector), and `LLM_BASE_URL` / `LLM_MODEL` pointing at a running
local OpenAI-compatible server. The embedding model (`EMBED_MODEL_ID`) downloads
automatically on first run.

### 5. Run the app
```bash
uv run streamlit run src/corpus_rag/app.py
```

## Testing

### Offline (no services needed)
Fast unit suite — settings, adapters, dimension contract, indexing wiring, query
grounding gate, eval metrics. Runs anywhere:
```bash
uv run pytest -m "not live"
```

### Live (needs real services)
Exercises the real embedder + Postgres/pgvector + local LLM end to end. **Ingest a
corpus first** (live tests self-skip on an empty corpus):
```bash
uv run corpus-rag ingest
uv run pytest -m live -rs          # -rs prints the reason for any skip
```
Notes:
- First run downloads the embedding + reranker models (~600 MB) and needs enough
  RAM to load them.
- Some live tests **skip by design** (not failures), e.g. a rerank-vs-cosine
  reordering demo skips when the corpus is too small to expect disagreement, and
  corpus-dependent tests skip if nothing has been ingested yet. `-rs` shows the
  exact reason for each.

### Full acceptance harness
Runs every spec §7 + §2A grounding check end-to-end and writes a PASS/FAIL report:
```bash
uv run python scripts/acceptance.py --report report.md
```

### Retrieval / grounding eval (optional)
Precision/recall/MRR/nDCG against a gold set, plus reference-free grounding
faithfulness. See `./scripts/eval.py`:
```bash
uv run python scripts/eval.py --qrels tests/eval/qrels.example.json --reference-free
```
```bash
uv run python scripts/eval.py --auto-generate 25   # build a gold set from the corpus
```
