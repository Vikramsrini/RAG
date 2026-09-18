"""
src/rag/ingestion/loaders.py
─────────────────────────────
Multi-format document loaders. Each loader returns a list of `Document`
dataclasses containing raw text and rich metadata.

Supported formats:
  • PDF  — via PyMuPDF (fitz)
  • TXT  — with charset detection
  • MD   — Markdown (plain text passthrough)
  • DOCX — via python-docx
  • HTML — via BeautifulSoup4
  • CSV  — via pandas
"""

from __future__ import annotations

import csv
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import chardet

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Document:
    """A loaded document with raw text and associated metadata."""

    text: str
    metadata: dict = field(default_factory=dict)

    # Convenience properties
    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> int | None:
        return self.metadata.get("page")

    def __repr__(self) -> str:  # noqa: D401
        preview = self.text[:60].replace("\n", " ")
        return f"Document(source={self.source!r}, chars={len(self.text)}, preview={preview!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────


class BaseLoader(ABC):
    """Abstract loader — subclass and implement `load(path)`."""

    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """Load a file and return a list of Document objects."""

    def _base_metadata(self, path: Path) -> dict:
        """Common metadata fields present in every document."""
        return {
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size if path.exists() else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PDF Loader
# ─────────────────────────────────────────────────────────────────────────────


class PDFLoader(BaseLoader):
    """
    Loads PDF files page-by-page using PyMuPDF.

    Each page becomes its own Document, preserving page number metadata.
    Tables are extracted as text blocks; images are skipped.
    """

    def load(self, path: Path) -> list[Document]:
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError("Install PyMuPDF: pip install pymupdf") from e

        documents: list[Document] = []
        base_meta = self._base_metadata(path)

        with fitz.open(str(path)) as pdf:
            total_pages = len(pdf)
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text")  # plain text, preserving layout
                if not text.strip():
                    logger.debug("Skipping empty page %d/%d in %s", page_num, total_pages, path.name)
                    continue

                documents.append(
                    Document(
                        text=text,
                        metadata={
                            **base_meta,
                            "page": page_num,
                            "total_pages": total_pages,
                            "doc_type": "pdf",
                        },
                    )
                )

        logger.info("PDF: loaded %d pages from '%s'", len(documents), path.name)
        return documents


# ─────────────────────────────────────────────────────────────────────────────
# TXT Loader
# ─────────────────────────────────────────────────────────────────────────────


class TXTLoader(BaseLoader):
    """
    Loads plain-text files with automatic charset detection.

    Falls back to UTF-8 → latin-1 if chardet is uncertain.
    """

    def load(self, path: Path) -> list[Document]:
        raw_bytes = path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"
        confidence = detected.get("confidence", 0.0)

        if confidence < 0.7:
            logger.debug(
                "Low charset confidence (%.0f%%) for %s, falling back to utf-8",
                confidence * 100,
                path.name,
            )
            encoding = "utf-8"

        try:
            text = raw_bytes.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            text = raw_bytes.decode("latin-1", errors="replace")

        meta = self._base_metadata(path)
        meta.update({"doc_type": "txt", "detected_encoding": encoding})
        logger.info("TXT: loaded %d chars from '%s'", len(text), path.name)
        return [Document(text=text, metadata=meta)]


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Loader
# ─────────────────────────────────────────────────────────────────────────────


class MarkdownLoader(TXTLoader):
    """
    Loads Markdown files as plain text.

    Inherits charset-safe loading from TXTLoader and overrides doc_type.
    Markdown syntax (headers, bold, links) is intentionally preserved as-is —
    the cleaner pipeline handles stripping if needed.
    """

    def load(self, path: Path) -> list[Document]:
        docs = super().load(path)
        for doc in docs:
            doc.metadata["doc_type"] = "markdown"
        return docs


# ─────────────────────────────────────────────────────────────────────────────
# DOCX Loader
# ─────────────────────────────────────────────────────────────────────────────


class DOCXLoader(BaseLoader):
    """
    Loads Word .docx files using python-docx.

    Extracts paragraph text with run-level formatting stripped.
    Tables are extracted row-by-row as pipe-delimited text.
    """

    def load(self, path: Path) -> list[Document]:
        try:
            import docx
        except ImportError as e:
            raise ImportError("Install python-docx: pip install python-docx") from e

        doc = docx.Document(str(path))
        parts: list[str] = []

        # Paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)

        # Tables — flatten to pipe-delimited rows
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))

        text = "\n".join(parts)
        meta = self._base_metadata(path)
        meta.update(
            {
                "doc_type": "docx",
                "paragraphs": len(doc.paragraphs),
                "tables": len(doc.tables),
            }
        )
        logger.info("DOCX: loaded %d paragraphs from '%s'", len(doc.paragraphs), path.name)
        return [Document(text=text, metadata=meta)]


# ─────────────────────────────────────────────────────────────────────────────
# HTML Loader
# ─────────────────────────────────────────────────────────────────────────────


class HTMLLoader(BaseLoader):
    """
    Loads HTML files using BeautifulSoup4.

    Removes scripts, styles, nav, footer, and header elements before
    extracting the main readable text.
    """

    _NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript"]

    def load(self, path: Path) -> list[Document]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError("Install beautifulsoup4: pip install beautifulsoup4 lxml") from e

        raw = path.read_bytes()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"

        soup = BeautifulSoup(raw, "lxml", from_encoding=encoding)

        # Remove noise elements
        for tag in self._NOISE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # Extract title
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        text = soup.get_text(separator="\n", strip=True)

        meta = self._base_metadata(path)
        meta.update({"doc_type": "html", "title": title})
        logger.info("HTML: extracted %d chars from '%s'", len(text), path.name)
        return [Document(text=text, metadata=meta)]


# ─────────────────────────────────────────────────────────────────────────────
# CSV Loader
# ─────────────────────────────────────────────────────────────────────────────


class CSVLoader(BaseLoader):
    """
    Loads CSV files using pandas, converting each row to a human-readable
    key:value sentence for embedding.

    Example row  →  "Name: Alice | Age: 30 | City: Paris"
    """

    def __init__(self, max_rows: int = 5000) -> None:
        self.max_rows = max_rows

    def load(self, path: Path) -> list[Document]:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("Install pandas: pip install pandas") from e

        # Detect encoding
        raw = path.read_bytes()
        encoding = chardet.detect(raw).get("encoding") or "utf-8"

        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding, nrows=self.max_rows)
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", nrows=self.max_rows)

        # Convert each row to a readable sentence
        parts: list[str] = []
        for _, row in df.iterrows():
            items = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            if items:
                parts.append(" | ".join(items))

        text = "\n".join(parts)
        meta = self._base_metadata(path)
        meta.update(
            {
                "doc_type": "csv",
                "rows": len(df),
                "columns": list(df.columns),
            }
        )
        logger.info("CSV: loaded %d rows from '%s'", len(df), path.name)
        return [Document(text=text, metadata=meta)]


# ─────────────────────────────────────────────────────────────────────────────
# Loader registry & factory
# ─────────────────────────────────────────────────────────────────────────────


_LOADER_REGISTRY: dict[str, type[BaseLoader]] = {
    ".pdf": PDFLoader,
    ".txt": TXTLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".docx": DOCXLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
    ".csv": CSVLoader,
}


def get_loader(path: Path) -> BaseLoader:
    """Return the appropriate loader for the given file extension."""
    ext = path.suffix.lower()
    loader_cls = _LOADER_REGISTRY.get(ext)
    if loader_cls is None:
        supported = ", ".join(_LOADER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported file type '{ext}' for '{path.name}'. "
            f"Supported extensions: {supported}"
        )
    return loader_cls()


def load_document(path: Path) -> list[Document]:
    """Convenience function: load any supported file and return Documents."""
    loader = get_loader(path)
    return loader.load(path)
