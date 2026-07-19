# OmniCorpus

**Grounded RAG Explorer** — clinical RAG over multiple sources: semantic retrieval
+ **grounded** generation over a heterogeneous document corpus, with a Streamlit UI
that shows the answer **beside its verbatim sources**.

## PHI / safety note

This demo is not intended for PHI, diagnosis, treatment recommendations, or
clinical decision support. The committed fixtures are synthetic and non-PHI. Real
clinical corpora should be used only in an appropriately secured environment with
access controls, audit logging, retention controls, and organization-approved
data handling.

## Why verbatim sources, always (the point of this app)

In healthcare, an answer alone is not enough. A reader has to be able to check it.

This app never shows a generated answer by itself. Every answer is displayed
**next to the exact, word-for-word source text it came from** — the same text that
was stored in the database and retrieved for the question. Nothing is paraphrased,
summarized, or re-written between storage and display.

In plain terms (and stated precisely — this is a narrow, specific guarantee, not
a claim that the model is sealed off from all general knowledge):

- **Specific clinical claims must come from the retrieved passages.** Any fact,
  value, dose, threshold, drug name, or recommendation has to trace to a chunk.
  The prompt forbids sourcing *specific* clinical facts from the model's training.
  General, non-specific reasoning (common definitions, plain-language connective
  text) may still be used to phrase the answer — so this is not a claim that the
  model uses *zero* non-corpus knowledge.
- **No grounding → an error/abstention, never a fallback.** If retrieval finds no
  chunk meeting the similarity threshold, the app returns a fixed abstention
  message ("Insufficient grounding in the corpus to answer.") and shows the
  (possibly empty) source list. It does **not** fall back to the summarizing
  model's own healthcare knowledge to manufacture an answer.
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
Two backends, switchable — run everything in Docker, or point at your own
LAN/remote Postgres + LLM. Embeddings + reranker always run in the host Python
process either way.

**Option A — fully containerized (DB + LLM):**
```bash
make stack                  # docker compose --profile llm up -d
make wait                   # block until the LLM has downloaded + loaded (/health)
make backend-docker         # point .env at the container endpoints
```
The `llm` service is llama.cpp's server with a 0.5B Qwen GGUF (CPU; auto-
downloaded, OpenAI API at `:8080/v1`). First `make stack` pulls the model, so
`make wait` (or `docker compose --profile llm up -d --wait`) blocks on its
`/health` until inference is actually ready. Swap the `-hf` model in
`./docker-compose.yml` for something larger if you have the cores.

**Option B — your own services (LAN/remote):**
- **Postgres + pgvector** — bundled container (`make db`) or your own server.
- **A local OpenAI-compatible LLM** (llama.cpp / vLLM / Ollama) on `/v1`.
- Set the endpoints in `.env` (see step 3). `make backend-lan` restores a
  previously saved `.env` from `.env.bak`.

`make help` lists all targets (`db`, `stack`, `down`, `ingest`, `app`, `test`,
`test-live`).

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

### 4. Run the app and ingest from the GUI
The app starts against an **empty database** — it creates the store table on
launch, so no prior ingest is required. Loading documents is a first-class GUI
action: open the app, use the sidebar **"Manage Documents"** uploader (PDF, DOCX,
PPTX, HTML, MD) to ingest at runtime, and remove a document by selecting its row
and clicking **"Delete selected"**. (The sidebar opens automatically when the
corpus is empty and stays collapsed once it has documents.)
```bash
uv run streamlit run src/corpus_rag/app.py
```
To make `.env` work you only need a reachable `PG_CONN_STR` (Postgres + pgvector)
and `LLM_BASE_URL` / `LLM_MODEL` pointing at a running local OpenAI-compatible
server. The embedding model (`EMBED_MODEL_ID`) downloads automatically on first run.

#### Optional: batch ingest from the CLI
For scripted/CI loads (or to seed the bundled sample), ingest the configured
`CORPUS_SOURCES` from the command line — the same pipeline as the GUI:
```bash
uv run corpus-rag ingest          # add --reset to recreate the table
```
A tiny **synthetic sample** ships in three formats —
`./tests/data/sample-clinical-guideline.{pdf,docx,html}` — so a fresh clone has
something to ingest (and the mixed-format PDF+DOCX+HTML claim is backed by real
fixtures). It is generated, non-PHI, common-knowledge text; regenerate with
`uv run python scripts/make_sample_corpus.py`. The default `CORPUS_SOURCES` points
at `tests/data/**/*`. Real/licensed corpora are gitignored — drop them in
`./tests/data/` (or just upload via the GUI).

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
Precision/recall/MRR/nDCG against a gold set, plus label-free grounding signals
(abstention rate, LLM-judge faithfulness, lexical citation coverage). See
`./scripts/eval.py`:
```bash
uv run python scripts/eval.py --qrels tests/eval/qrels.example.json --reference-free
```
```bash
uv run python scripts/eval.py --auto-generate 25   # build a gold set from the corpus
```
**Honest limits:** auto-generated qrels test *self-retrievability* (the question is
written from the chunk it then retrieves), and the faithfulness judge defaults to
the **same** local model as the generator (self-grading). Treat these as smoke
signals; use a hand-authored gold set and a separate judge model for real numbers.

## Troubleshooting & hardware

**`[RapidOCR] Using CPU device` on an Apple-Silicon / Metal box — why not the GPU?**
Expected, not a misconfiguration. RapidOCR (the OCR engine Docling uses for
scanned PDF pages) runs on its torch backend, whose device selector only checks
`torch.cuda.is_available()`. Apple's GPU is **MPS**, not CUDA, so the check fails
and it falls back to CPU; even forced onto MPS, several OCR ops lack MPS kernels
and silently drop back to CPU. "Auto-GPU" in most ML libs means CUDA — Metal
support is opt-in and, for RapidOCR, absent. OCR is a one-time **ingest** cost, so
CPU is usually fine. Options:
- Born-digital corpus (text layers, no scans)? Set `OCR_ON=false` — RapidOCR never
  loads, the log disappears, ingest is faster, and text still extracts.
- Need OCR but want it quiet? `logging.getLogger("RapidOCR").setLevel("WARNING")`.
- Real OCR GPU acceleration realistically needs a CUDA host.

**Does anything use the Apple GPU, then?** Yes — the embedder and cross-encoder
reranker (sentence-transformers / torch) are a separate path; Haystack's device
auto-resolve prefers `cuda → mps → cpu`, so on Apple Silicon they use **MPS**.
That's the repeated, latency-critical work (every query); OCR is the one-time
ingest cost. The GPU win is where it matters.

## FAQ (likely reviewer questions)

- **How do you stop the LLM inventing clinical facts?** Two layers. (1) A hard
  retrieval gate: if no chunk scores at/above `MIN_SCORE` the app abstains
  *before* the generator runs (`run_query` gates between retrieve and generate) —
  the model is never called without grounding. (2) A grounding-only prompt:
  specific clinical claims must come from the retrieved chunks; general reasoning
  is allowed, parametric specifics are not. See "Why verbatim sources".
- **Why is `MIN_SCORE` 0.35 by default, not 0?** Fail-safe: at 0.0 the gate is a
  no-op (retriever always returns `top_k`, generator always runs) and grounding
  rests on the prompt alone. Non-zero makes the mechanical gate active by default;
  `build_*_engine` warns if you set it to 0. Tune the exact floor per corpus/model
  from the eval harness.
- **Why a general embedding model, not a clinical/biomedical one?** Domain safety
  here comes from grounding in the uploaded corpus, not from a domain-tuned model.
  Corpus scope is set at runtime (cardiology today, nephrology tomorrow), so the
  app stays domain-agnostic and is measured *relative to the uploaded corpus*.
- **Why JSON qrels, not YAML?** No new dependency, and it matches the existing
  `CORPUS_SOURCES` JSON convention.
- **Why does `pytest -m live` show skips?** By design (`-rs` prints reasons):
  corpus-dependent tests skip on an empty store; the rerank-vs-cosine demo skips
  when the corpus is too small to expect disagreement. None are failures.
- **Live vs offline tests?** Offline (`-m "not live"`, run in CI) needs no
  services — pure logic with fakes. Live (`-m live`) needs Postgres+pgvector + a
  local LLM and is run manually. CI never requires those.
- **Is the committed corpus real patient data?** No — synthetic, non-PHI fixtures
  (`./tests/data/sample-clinical-guideline.{pdf,docx,html}`). See the PHI/safety
  note above.
