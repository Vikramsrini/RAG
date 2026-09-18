"""
src/rag/ingestion/pipeline.py
──────────────────────────────
Orchestrates the full ingestion flow:

  File/Directory  →  Load  →  Clean  →  Chunk  →  List[Chunk]

Features:
  • Batch processing of entire directories (recursive or flat)
  • Per-file error isolation (one bad file doesn't stop the batch)
  • Skip-already-ingested logic via SHA-256 fingerprint registry
  • Rich progress bars in CLI mode
  • Structured logging throughout
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from rag.ingestion.chunker import Chunk, TokenAwareChunker, build_chunker_from_settings
from rag.ingestion.cleaner import TextCleaner, build_cleaner_from_settings
from rag.ingestion.loaders import Document, load_document

logger = logging.getLogger(__name__)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint registry — skip already-ingested files
# ─────────────────────────────────────────────────────────────────────────────


class _FingerprintRegistry:
    """Persists SHA-256 fingerprints of ingested files to a JSON file."""

    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path
        self._data: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def fingerprint(self, path: Path) -> str:
        """Compute SHA-256 of file content (first 1 MB is sufficient)."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            h.update(f.read(1024 * 1024))
        return h.hexdigest()

    def is_seen(self, path: Path) -> bool:
        fp = self.fingerprint(path)
        return self._data.get(str(path)) == fp

    def mark_seen(self, path: Path) -> None:
        self._data[str(path)] = self.fingerprint(path)
        self._save()


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class IngestionPipeline:
    """
    End-to-end ingestion pipeline.

    Usage::

        pipeline = IngestionPipeline()
        chunks = pipeline.run(Path("data/documents"))

    Parameters
    ----------
    cleaner : TextCleaner | None
        Override the default cleaner (built from settings if None).
    chunker : TokenAwareChunker | None
        Override the default chunker (built from settings if None).
    show_progress : bool
        Show Rich progress bars (useful in CLI mode).
    """

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        chunker: TokenAwareChunker | None = None,
        show_progress: bool = True,
    ) -> None:
        from rag.config import get_settings

        self._settings = get_settings()
        self._cleaner = cleaner or build_cleaner_from_settings()
        self._chunker = chunker or build_chunker_from_settings()
        self._show_progress = show_progress

        # Fingerprint registry lives next to the vectorstore
        registry_path = self._settings.paths.vectorstore_dir / ".ingestion_registry.json"
        self._registry = _FingerprintRegistry(registry_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        path: Path | str,
        force: bool = False,
    ) -> list[Chunk]:
        """
        Ingest a file or directory.

        Parameters
        ----------
        path : Path | str
            A single file or a directory to scan recursively.
        force : bool
            If True, re-ingest files even if already seen.

        Returns
        -------
        list[Chunk]
            All chunks produced from the ingested documents.
        """
        path = Path(path)
        files = list(self._collect_files(path))

        if not files:
            logger.warning("No supported files found at '%s'", path)
            return []

        logger.info("Found %d file(s) to process", len(files))
        all_chunks: list[Chunk] = []
        stats = {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0}

        if self._show_progress:
            all_chunks, stats = self._run_with_progress(files, force, stats)
        else:
            all_chunks, stats = self._run_silent(files, force, stats)

        self._print_summary(stats, all_chunks)
        return all_chunks

    def run_file(self, path: Path, force: bool = False) -> list[Chunk]:
        """Ingest a single file."""
        path = Path(path)
        if not force and self._registry.is_seen(path):
            logger.info("Skipping already-ingested file: %s", path.name)
            return []

        try:
            chunks = self._process_file(path)
            self._registry.mark_seen(path)
            return chunks
        except Exception as exc:
            logger.error("Failed to ingest '%s': %s", path.name, exc, exc_info=True)
            return []

    # ── Internal processing ───────────────────────────────────────────────────

    def _process_file(self, path: Path) -> list[Chunk]:
        """Load → Clean → Chunk a single file."""
        # 1. Load
        logger.debug("Loading: %s", path.name)
        docs: list[Document] = load_document(path)

        if not docs:
            logger.warning("No content extracted from '%s'", path.name)
            return []

        # 2. Clean — in-place mutation of doc.text
        cleaned_docs: list[Document] = []
        for doc in docs:
            cleaned_text = self._cleaner.clean(doc.text)
            if cleaned_text:
                doc.text = cleaned_text
                cleaned_docs.append(doc)
            else:
                logger.debug("Dropping empty doc after cleaning: %s (page %s)", path.name, doc.page)

        if not cleaned_docs:
            logger.warning("All content filtered out after cleaning: %s", path.name)
            return []

        # 3. Chunk
        chunks = self._chunker.chunk_documents(cleaned_docs)
        logger.debug(
            "'%s' → %d doc(s) → %d chunk(s)",
            path.name, len(cleaned_docs), len(chunks),
        )
        return chunks

    def _collect_files(self, path: Path) -> Iterator[Path]:
        """Yield all supported files under path."""
        exts = set(self._settings.ingestion.supported_extensions)
        if path.is_file():
            if path.suffix.lower() in exts:
                yield path
        elif path.is_dir():
            glob = "**/*" if self._settings.ingestion.recursive else "*"
            for p in sorted(path.glob(glob)):
                if p.is_file() and p.suffix.lower() in exts:
                    yield p
        else:
            logger.error("Path does not exist: %s", path)

    def _run_with_progress(self, files, force, stats):
        all_chunks = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Ingesting documents", total=len(files))
            for path in files:
                progress.update(task, description=f"[bold blue]{path.name[:40]}")
                if not force and self._settings.ingestion.skip_existing and self._registry.is_seen(path):
                    stats["skipped"] += 1
                    progress.advance(task)
                    continue
                try:
                    chunks = self._process_file(path)
                    self._registry.mark_seen(path)
                    all_chunks.extend(chunks)
                    stats["processed"] += 1
                    stats["total_chunks"] += len(chunks)
                except Exception as exc:
                    logger.error("Failed '%s': %s", path.name, exc)
                    stats["failed"] += 1
                progress.advance(task)
        return all_chunks, stats

    def _run_silent(self, files, force, stats):
        all_chunks = []
        for path in files:
            if not force and self._settings.ingestion.skip_existing and self._registry.is_seen(path):
                stats["skipped"] += 1
                continue
            try:
                chunks = self._process_file(path)
                self._registry.mark_seen(path)
                all_chunks.extend(chunks)
                stats["processed"] += 1
                stats["total_chunks"] += len(chunks)
            except Exception as exc:
                logger.error("Failed '%s': %s", path.name, exc)
                stats["failed"] += 1
        return all_chunks, stats

    @staticmethod
    def _print_summary(stats: dict, chunks: list[Chunk]) -> None:
        table = Table(title="✅ Ingestion Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right", style="bold green")
        table.add_row("Files processed", str(stats["processed"]))
        table.add_row("Files skipped (cached)", str(stats["skipped"]))
        table.add_row("Files failed", str(stats["failed"]))
        table.add_row("Total chunks produced", str(stats["total_chunks"]))
        if chunks:
            avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
            table.add_row("Avg tokens/chunk", f"{avg_tokens:.0f}")
        console.print(table)
