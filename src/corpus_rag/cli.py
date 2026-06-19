"""Command-line entry point.

``corpus-rag ingest`` resolves every configured source (``CORPUS_SOURCES``) via
the adapter registry, then runs the indexing pipeline (Docling chunk -> embed ->
write) over them. ``--reset`` recreates the vector table first (dev reset).
``--discover-only`` lists resolved sources without loading models / touching the
database.
"""

from __future__ import annotations

from pathlib import Path

import typer

from corpus_rag.adapters import discover_all
from corpus_rag.adapters.base import Source
from corpus_rag.settings import get_settings

app = typer.Typer(help="Corpus RAG Explorer CLI.", no_args_is_help=True)


@app.callback()
def _main() -> None:
    """Corpus RAG Explorer CLI (keeps ``ingest`` as an explicit subcommand)."""


def _label(source: Source) -> str:
    """Human-readable label for a discovered source."""
    if isinstance(source, str | Path):
        return str(source)
    # ByteStream — show its provenance / mime type.
    url = source.meta.get("source_url") if source.meta else None
    return url or f"<bytes {source.mime_type or 'unknown'}>"


@app.command()
def ingest(
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Dev reset: recreate the vector table before ingesting.",
    ),
    discover_only: bool = typer.Option(
        False,
        "--discover-only",
        help="List resolved sources only; skip model load and DB writes.",
    ),
) -> None:
    """Discover all configured sources and index them into the vector store."""
    settings = get_settings()
    if not settings.corpus_sources:
        typer.echo("No CORPUS_SOURCES configured; nothing to ingest.")
        raise typer.Exit(code=1)

    sources = discover_all(settings.corpus_sources)
    typer.echo(f"Discovered {len(sources)} source(s):")
    for source in sources:
        typer.echo(f"  - {_label(source)}")

    if discover_only:
        return
    if not sources:
        typer.echo("Nothing to index.")
        raise typer.Exit(code=1)

    # Imported lazily: building the store + pipeline loads the embedding model
    # and connects to Postgres, which --discover-only callers must not pay for.
    from corpus_rag.document_store import build_document_store
    from corpus_rag.pipelines.indexing import build_indexing_pipeline, run_indexing

    store = build_document_store(settings, recreate_table=reset)
    pipeline = build_indexing_pipeline(store, settings)
    typer.echo("Indexing… (loading embedding model on first run)")
    run_indexing(pipeline, sources)
    typer.echo(f"Done. Store now holds {store.count_documents()} chunk(s).")


if __name__ == "__main__":
    app()
