"""
src/rag/vectorstore/store.py
─────────────────────────────
ChromaDB vector store wrapper.

Features:
  • Persistent local storage (no server needed)
  • Upsert with hash-based deduplication (re-ingesting same content is idempotent)
  • Batch upsert to avoid ChromaDB's per-call limits
  • Similarity search returning scored results
  • Collection statistics
  • Delete by source file
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHROMA_BATCH_SIZE = 500  # ChromaDB recommends < 5461 items per call


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single retrieved chunk with its similarity score."""

    id: str
    text: str
    score: float               # Cosine similarity (higher = more relevant)
    metadata: dict[str, Any]

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"SearchResult(score={self.score:.3f}, source={self.metadata.get('source', '?')!r}, preview={preview!r})"


# ─────────────────────────────────────────────────────────────────────────────
# ChromaVectorStore
# ─────────────────────────────────────────────────────────────────────────────


class ChromaVectorStore:
    """
    Persistent ChromaDB vector store.

    Parameters
    ----------
    persist_dir : Path | str
        Directory where ChromaDB files are stored.
    collection_name : str
        Name of the ChromaDB collection.
    distance_metric : str
        "cosine" | "l2" | "ip" (inner product).
    """

    def __init__(
        self,
        persist_dir: Path | str,
        collection_name: str = "rag_documents",
        distance_metric: str = "cosine",
    ) -> None:
        self._persist_dir = Path(persist_dir)
        self._collection_name = collection_name
        self._distance_metric = distance_metric
        self._client = None
        self._collection = None
        self._init_client()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_client(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as e:
            raise ImportError("Install ChromaDB: pip install chromadb") from e

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_metric},
        )

        logger.info(
            "ChromaDB initialized — collection '%s' has %d document(s)",
            self._collection_name,
            self._collection.count(),
        )

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> int:
        """
        Upsert chunks into the collection in batches.

        Upsert is idempotent — adding the same chunk ID twice updates in-place.

        Returns the number of items upserted.
        """
        if not ids:
            return 0

        total = 0
        for start in range(0, len(ids), _CHROMA_BATCH_SIZE):
            end = start + _CHROMA_BATCH_SIZE
            batch_ids = ids[start:end]
            batch_embs = embeddings[start:end]
            batch_docs = documents[start:end]
            batch_meta = metadatas[start:end]

            # Sanitize metadata — ChromaDB requires all values to be str/int/float/bool
            sanitized_meta = [self._sanitize_metadata(m) for m in batch_meta]

            self._collection.upsert(
                ids=batch_ids,
                embeddings=batch_embs,
                documents=batch_docs,
                metadatas=sanitized_meta,
            )
            total += len(batch_ids)
            logger.debug("Upserted batch of %d items (total so far: %d)", len(batch_ids), total)

        logger.info("Upserted %d chunks into '%s'", total, self._collection_name)
        return total

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Retrieve the top-k most similar chunks.

        Parameters
        ----------
        query_embedding : list[float]
            Embedding of the query text.
        top_k : int
            Number of results to return.
        where : dict | None
            ChromaDB metadata filter (e.g. {"doc_type": "pdf"}).
        score_threshold : float
            Minimum similarity score to include in results.

        Returns
        -------
        list[SearchResult]
            Ranked from highest to lowest similarity.
        """
        n_results = min(top_k, self._collection.count())
        if n_results == 0:
            logger.warning("Collection is empty — no results")
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        search_results: list[SearchResult] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for rid, doc, meta, dist in zip(ids, docs, metas, distances):
            # Convert distance to similarity score
            if self._distance_metric == "cosine":
                score = 1.0 - dist  # ChromaDB cosine distance → similarity
            elif self._distance_metric == "ip":
                score = dist        # Inner product is already a similarity
            else:
                score = 1.0 / (1.0 + dist)  # L2: invert distance

            if score >= score_threshold:
                search_results.append(
                    SearchResult(id=rid, text=doc, score=score, metadata=meta or {})
                )

        return search_results

    # ── Collection management ─────────────────────────────────────────────────

    def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks from a specific source file."""
        results = self._collection.get(where={"source": source_path})
        ids_to_delete = results["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info("Deleted %d chunks for source: %s", len(ids_to_delete), source_path)
        return len(ids_to_delete)

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self._collection_name)
        self._init_client()
        logger.warning("Collection '%s' cleared.", self._collection_name)

    def count(self) -> int:
        """Return the total number of chunks stored."""
        return self._collection.count()

    def stats(self) -> dict:
        """Return basic statistics about the collection."""
        n = self._collection.count()
        return {
            "collection": self._collection_name,
            "total_chunks": n,
            "persist_dir": str(self._persist_dir),
            "distance_metric": self._distance_metric,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        """
        ChromaDB only accepts str, int, float, bool values.
        Convert lists and None to safe types.
        """
        clean: dict = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                clean[k] = ", ".join(str(i) for i in v)
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = str(v)
        return clean


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_vectorstore() -> ChromaVectorStore:
    """Return a cached ChromaVectorStore configured from global Settings."""
    from rag.config import get_settings

    cfg = get_settings()
    return ChromaVectorStore(
        persist_dir=cfg.paths.vectorstore_dir,
        collection_name=cfg.vectorstore.collection_name,
        distance_metric=cfg.vectorstore.distance_metric,
    )
