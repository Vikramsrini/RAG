"""
cli/ingest.py
──────────────
CLI entry point for document ingestion.

Usage:
    python -m cli.ingest ./data/documents
    python -m cli.ingest ./report.pdf --force
    python -m cli.ingest ./data --stats
    python -m cli.ingest --clear
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=console,
                show_time=True,
                show_path=verbose,
                markup=True,
                rich_tracebacks=True,
            )
        ],
    )
    # Silence noisy third-party loggers
    for noisy in ["httpx", "httpcore", "urllib3", "sentence_transformers", "chromadb"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


@click.command()
@click.argument("path", required=False, default=None, type=click.Path())
@click.option("--force", "-f", is_flag=True, help="Re-ingest files even if already cached.")
@click.option("--stats", "-s", is_flag=True, help="Show knowledge base statistics and exit.")
@click.option("--clear", is_flag=True, help="Clear the entire knowledge base.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(
    path: str | None,
    force: bool,
    stats: bool,
    clear: bool,
    verbose: bool,
) -> None:
    """
    \b
    ╔══════════════════════════════════╗
    ║    RAG Document Ingestion CLI    ║
    ╚══════════════════════════════════╝

    Ingest documents into the local RAG knowledge base.

    \b
    Examples:
      # Ingest a directory
      python -m cli.ingest ./data/documents

      # Ingest a single file
      python -m cli.ingest ./report.pdf

      # Force re-ingest
      python -m cli.ingest ./data --force

      # Show stats
      python -m cli.ingest --stats

      # Clear knowledge base
      python -m cli.ingest --clear
    """
    _setup_logging(verbose)

    # Lazy import after logging setup
    from rag.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(show_progress=True)

    # ── Clear ────────────────────────────────────────────────────────────────
    if clear:
        console.print(Panel("[bold red]⚠️  Clearing knowledge base…[/bold red]"))
        if click.confirm("Are you sure? This will delete all ingested documents.", default=False):
            pipeline.clear_knowledge_base()
            console.print("[bold green]✅ Knowledge base cleared.[/bold green]")
        else:
            console.print("[yellow]Aborted.[/yellow]")
        return

    # ── Stats ────────────────────────────────────────────────────────────────
    if stats:
        s = pipeline.stats()
        table = Table(title="📊 Knowledge Base Stats", header_style="bold cyan")
        table.add_column("Property", style="dim")
        table.add_column("Value", style="bold green")
        for k, v in s.items():
            table.add_row(str(k), str(v))
        console.print(table)
        return

    # ── Ingest ───────────────────────────────────────────────────────────────
    if not path:
        console.print("[red]Error:[/red] Please provide a path to ingest.")
        console.print("Run with --help for usage information.")
        sys.exit(1)

    ingest_path = Path(path)
    if not ingest_path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: [bold]{ingest_path}[/bold]")
        sys.exit(1)

    console.print(
        Panel(
            Text.assemble(
                ("📂 Ingesting: ", "bold cyan"),
                (str(ingest_path.resolve()), "bold white"),
                ("\n🔄 Force re-ingest: ", "dim"),
                ("Yes" if force else "No", "bold yellow" if force else "dim"),
            ),
            title="[bold blue]RAG Ingestion[/bold blue]",
            border_style="blue",
        )
    )

    n = pipeline.ingest(ingest_path, force=force)
    console.print(f"\n[bold green]✅ Done! {n} chunks stored in the knowledge base.[/bold green]")


if __name__ == "__main__":
    main()
