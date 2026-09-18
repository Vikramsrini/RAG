"""
src/rag/ingestion/chunker.py
─────────────────────────────
Token-aware recursive text splitter.

Why token-aware?
  Character-based splitting is inaccurate — "hello" is 1 token but
  "Supercalifragilistic" is 5. Using tiktoken we get exact token counts
  that match what the embedding model actually sees.

Algorithm:
  1. Try to split on the highest-priority separator (e.g. "\\n\\n").
  2. If any resulting piece is still too large, recurse with the next separator.
  3. Merge adjacent small pieces up to chunk_size with chunk_overlap bridging.
  4. Discard chunks shorter than min_chunk_size tokens.

Output:
  List of `Chunk` objects, each with text, token count, chunk index,
  and inherited document metadata.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import tiktoken

from rag.ingestion.loaders import Document

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A text chunk ready for embedding."""

    text: str
    token_count: int
    chunk_index: int               # Index within the source document
    doc_id: str                    # Stable hash of source path + page
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Unique chunk ID derived from source + position."""
        return f"{self.doc_id}__chunk_{self.chunk_index:04d}"

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(id={self.id!r}, tokens={self.token_count}, "
            f"preview={preview!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TokenAwareChunker
# ─────────────────────────────────────────────────────────────────────────────


class TokenAwareChunker:
    """
    Recursively splits text into token-bounded, overlapping chunks.

    Parameters
    ----------
    chunk_size : int
        Maximum number of tokens per chunk.
    chunk_overlap : int
        Number of tokens to overlap between consecutive chunks.
    min_chunk_size : int
        Chunks with fewer tokens than this are discarded.
    encoding_name : str
        tiktoken encoding (e.g. "cl100k_base" for GPT-4 / text-embedding-3).
    separators : list[str]
        Ordered list of split candidates (highest priority first).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        encoding_name: str = "cl100k_base",
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        self._enc = tiktoken.get_encoding(encoding_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Chunk a single Document into Chunk objects."""
        doc_id = self._stable_id(doc)
        raw_chunks = self._split(doc.text, self.separators)
        chunks: list[Chunk] = []

        for idx, text in enumerate(raw_chunks):
            token_count = self._count_tokens(text)
            if token_count < self.min_chunk_size:
                logger.debug("Skipping short chunk (%d tokens)", token_count)
                continue

            chunks.append(
                Chunk(
                    text=text,
                    token_count=token_count,
                    chunk_index=idx,
                    doc_id=doc_id,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "token_count": token_count,
                        "doc_id": doc_id,
                    },
                )
            )

        logger.debug(
            "Chunked '%s' → %d chunks (avg %.0f tokens)",
            doc.source,
            len(chunks),
            sum(c.token_count for c in chunks) / max(len(chunks), 1),
        )
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Chunk a list of Documents, flattening all results."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # ── Token utilities ───────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self._enc.encode(text, disallowed_special=()))

    def _encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def _decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)

    # ── Recursive splitting ───────────────────────────────────────────────────

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the separator hierarchy, then merge
        small pieces up to chunk_size with overlap.
        """
        if not text.strip():
            return []

        token_count = self._count_tokens(text)
        if token_count <= self.chunk_size:
            return [text]

        # Try each separator in priority order
        separator = ""
        remaining_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                remaining_separators = separators[i + 1:]
                break

        # Split on the chosen separator
        if separator:
            splits = text.split(separator)
        else:
            # Last resort: split by tokens directly
            return self._split_by_tokens(text)

        # Recursively chunk pieces that are still too large
        good_splits: list[str] = []
        for piece in splits:
            piece = piece.strip()
            if not piece:
                continue
            if self._count_tokens(piece) > self.chunk_size:
                good_splits.extend(self._split(piece, remaining_separators or [""]))
            else:
                good_splits.append(piece)

        # Merge small pieces into final chunks with overlap
        return self._merge_with_overlap(good_splits, separator)

    def _split_by_tokens(self, text: str) -> list[str]:
        """Hard split on token boundaries when no separator works."""
        tokens = self._encode(text)
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._decode(chunk_tokens))
            start = end - self.chunk_overlap
            if start >= end:
                break
        return chunks

    def _merge_with_overlap(self, splits: list[str], separator: str) -> list[str]:
        """
        Greedily merge adjacent splits into chunks no larger than chunk_size,
        prepending the last `chunk_overlap` tokens from the previous chunk.
        """
        chunks: list[str] = []
        current_tokens: list[int] = []
        current_len = 0

        for split in splits:
            split_tokens = self._encode(split)
            split_len = len(split_tokens)

            if current_len + split_len > self.chunk_size and current_tokens:
                # Emit the current chunk
                chunks.append(self._decode(current_tokens))
                # Keep overlap tokens for the next chunk
                if self.chunk_overlap > 0:
                    current_tokens = current_tokens[-self.chunk_overlap:]
                    current_len = len(current_tokens)
                else:
                    current_tokens = []
                    current_len = 0

            current_tokens.extend(split_tokens)
            current_len += split_len

        # Emit the last chunk
        if current_tokens:
            chunks.append(self._decode(current_tokens))

        return chunks

    # ── Stable ID ─────────────────────────────────────────────────────────────

    @staticmethod
    def _stable_id(doc: Document) -> str:
        """
        Derive a stable, content-independent ID from source path + page.
        Uses SHA-256 truncated to 12 hex chars.
        """
        key = f"{doc.metadata.get('source', 'unknown')}::{doc.metadata.get('page', 0)}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────


def build_chunker_from_settings() -> TokenAwareChunker:
    """Build a TokenAwareChunker configured from global Settings."""
    from rag.config import get_settings

    cfg = get_settings().chunking
    return TokenAwareChunker(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        min_chunk_size=cfg.min_chunk_size,
        encoding_name=cfg.tokenizer_encoding,
        separators=cfg.separators,
    )
