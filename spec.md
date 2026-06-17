# Spec: Corpus RAG Explorer

> Zenflow **Spec First** artifact. Single-slice, runnable RAG app over a heterogeneous
> document corpus, with a minimal Streamlit GUI for query → response → ranked sources.
> Scope is deliberately narrow so it ships as one task; extensions are listed as non-goals.

## 1. Goal

Expose a corpus of documents — arriving from **different sources** and in **different
formats** — through semantic retrieval and grounded generation. A user types a query and
sees (a) the query, (b) an LLM-generated response grounded in retrieved context, and
(c) the top 10 source chunks ranked by semantic distance, each shown as a one-line teaser
that expands to the full section of prose plus its provenance.

## 2. Locked stack (decision record)

| Concern | Choice | Package |
| --- | --- | --- |
| Runtime | Python 3.12, managed with `uv` | — |
| Retrieval orchestration | Haystack 2.x (explicit, named pipeline components) | `haystack-ai` |
| Multi-format ingestion | Docling via Haystack converter | `docling-haystack`, `docling` |
| Embeddings | **Local** sentence-transformers | `sentence-transformers` (via Haystack embedders) |
| Vector store | PostgreSQL + pgvector | `pgvector-haystack`, `psycopg` |
| Generation | Local OpenAI-compatible endpoint | `haystack-ai` (`OpenAIGenerator`) |
| GUI | Streamlit, click-to-expand | `streamlit` |

No FastAPI: Streamlit is the entire presentation + interaction layer.

## 3. Architecture

Two Haystack pipelines plus a thin source-adapter layer and a Streamlit front end.

```
sources ──> SourceAdapter registry ──> [INDEXING PIPELINE]
                                          DoclingConverter (DOC_CHUNKS, HybridChunker)
                                            -> SentenceTransformersDocumentEmbedder
                                            -> DocumentWriter -> PgvectorDocumentStore

query ──> [QUERY PIPELINE]
            SentenceTransformersTextEmbedder
              -> PgvectorEmbeddingRetriever (top_k=10)
              -> PromptBuilder -> OpenAIGenerator (local) -> response
          (retriever documents also returned for the source list)

Streamlit ──> runs QUERY PIPELINE, renders query + response + 10 expanders
```

### 3.1 Source adapters
Docling natively parses PDF, DOCX, PPTX, HTML, Markdown, and more, so format coverage is
largely Docling's job. The adapter layer handles **origin**, not format: a small registry
that resolves each configured source (local path/glob, URL, future object store) to a list
of file paths or `ByteStream`s handed to `DoclingConverter.run(paths=...)`. New origins =
new adapter implementing a single `discover() -> list[source]` method.

### 3.2 Indexing pipeline
- `DoclingConverter(export_type=ExportType.DOC_CHUNKS, chunker=HybridChunker(tokenizer=EMBED_MODEL_ID))`
  — emits one Haystack `Document` per chunk, carrying Docling provenance metadata
  (source ref, page, heading path) in `Document.meta`.
- `SentenceTransformersDocumentEmbedder(model=EMBED_MODEL_ID)` — fills `Document.embedding`.
- `DocumentWriter(document_store, policy=DuplicatePolicy.OVERWRITE)` — idempotent re-ingest.
- Wiring: `converter -> embedder -> writer`.

### 3.3 Query pipeline
- `SentenceTransformersTextEmbedder(model=EMBED_MODEL_ID)` — **must be the same model** as
  the document embedder.
- `PgvectorEmbeddingRetriever(document_store=..., top_k=10)`.
- `PromptBuilder` — RAG prompt template injecting retrieved chunk contents + sources.
- `OpenAIGenerator` pointed at a local OpenAI-compatible `base_url` (e.g. a llama.cpp /
  vLLM server). Temperature 0 for reproducible answers.
- Wiring: `text_embedder.embedding -> retriever.query_embedding`; retriever docs ->
  prompt builder -> generator. The pipeline returns BOTH the generated answer and the
  retriever's `documents` list (the latter drives the source display).

## 4. Persistence & data model

`PgvectorDocumentStore` owns the schema. Configure:
- `embedding_dimension = EMBEDDING_DIM` — **hard contract**: must equal the embedding
  model's output dimension (e.g. 768 for `all-mpnet-base-v2` / `bge-base-en-v1.5`,
  384 for `all-MiniLM-L6-v2`, 1024 for `bge-large-en-v1.5`).
- `vector_function = "cosine_similarity"` (semantic distance ranking).
- `search_strategy = "hnsw"` (ANN index).
- `recreate_table = False` in normal runs; `True` only for an explicit dev reset.

Each stored chunk row carries: `content` (the full section prose — this is the expander
payload AND the citation unit), `embedding`, and `meta` (source ref, page, heading path).
The display "first line" is derived from `content`, not stored separately.

Provenance note: because the displayed/expanded chunk and the embedded/retrieved chunk are
the same unit with source metadata attached, the UI's "show original prose" and audit-grade
source traceability are satisfied by one field. Keep it that way.

## 5. GUI contract (Streamlit)

Single page:
1. Query input (`st.text_input` or `st.chat_input`).
2. On submit, run the query pipeline.
3. Render `st.markdown` of the query, then the generated **response**.
4. Render a "Sources" section: iterate the 10 retriever documents **in returned order**
   (already ranked by semantic distance). For each:
   - `st.expander(label=first_line)` where `first_line = doc.content.strip().splitlines()[0]`
     truncated to ~120 chars.
   - Inside the expander: the full `doc.content`, plus `doc.meta` (source, page, heading)
     and the similarity `doc.score`.

Click-to-expand (accepted in place of hover). No JS, no custom components.

## 6. Configuration (env)

- `PG_CONN_STR` — `postgresql://USER:PASSWORD@HOST:PORT/DB` (percent-encode special chars
  in the password).
- `EMBED_MODEL_ID` — sentence-transformers model id; `EMBEDDING_DIM` derived from it.
- `TOP_K` — default 10.
- `LLM_BASE_URL`, `LLM_MODEL` — local OpenAI-compatible generator endpoint + model name.
- `CORPUS_SOURCES` — config block enumerating adapters and their roots/URLs.

## 7. Verification / acceptance criteria

1. `CREATE EXTENSION vector;` succeeds; store init creates the table + HNSW index.
2. Startup asserts `embedder.dimension == store.embedding_dimension`; mismatch fails fast.
3. Ingesting a mixed sample set (≥1 PDF, ≥1 DOCX, ≥1 HTML) writes N>0 chunks, each with a
   non-null embedding of length `EMBEDDING_DIM` and non-empty source metadata.
4. Re-ingesting the same documents does not increase chunk count (OVERWRITE idempotency).
5. A query returns exactly `min(10, corpus_size)` documents with non-increasing scores.
6. GUI shows query, a non-empty grounded response, and the ranked source list; expanding a
   row reveals the full section text and its source.
7. Determinism: same query + same corpus + temperature 0 yields a stable retrieval order.

## 8. Non-goals (defer to later tasks)

Auth / multi-user; hybrid or keyword retrieval (note `PgvectorKeywordRetriever` exists for a
future hybrid task); reranking; an evaluation harness (e.g. RAGAS); incremental/streaming
ingestion; conversation memory; horizontal scaling. Single-process Streamlit is sufficient.

## 9. Implementation plan

- [ ] Scaffold project with `uv`; pin all deps; `.env` + settings module.
- [ ] Bring up Postgres + pgvector (Docker); enable the extension.
- [ ] Document-store module: build `PgvectorDocumentStore` from config; assert dim contract.
- [ ] Source-adapter registry + local-path and URL adapters; folder-ingest CLI.
- [ ] Indexing pipeline (Docling → embedder → writer); run against sample corpus.
- [ ] Query pipeline (text embedder → retriever → prompt builder → local generator).
- [ ] Streamlit app: query box, response, 10 expanders with full-section + provenance.
- [ ] Mixed-format sample corpus + a script that runs the §7 acceptance checks.

## 10. References

- Haystack pgvector integration — https://docs.haystack.deepset.ai/docs/pgvectordocumentstore
- PgvectorEmbeddingRetriever — https://docs.haystack.deepset.ai/docs/pgvectorembeddingretriever
- Docling Haystack integration — https://haystack.deepset.ai/integrations/docling
- Docling + Haystack RAG example — https://docling-project.github.io/docling/examples/rag_haystack/
