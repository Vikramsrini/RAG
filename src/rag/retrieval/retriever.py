"""
src/rag/retrieval/retriever.py
───────────────────────────────
Dense retrieval with optional Maximum Marginal Relevance (MMR) re-ranking.

Dense retrieval:
  Embed the query → ANN search in ChromaDB → return top-k by cosine similarity.

MMR re-ranking (when enabled):
  1. Fetch a larger candidate pool (mmr_fetch_k results).
  2. Iteratively select chunks that maximise: λ·relevance − (1−λ)·redundancy
     where redundancy = max cosine similarity to already-selected chunks.
  3. This produces a diverse, non-redundant final set of top_k chunks.

MMR reference:
  Carbonell & Goldstein (1998), "The Use of MMR, Diversity-Based Reranking
  for Reordering Documents and Producing Summaries"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """A retrieved chunk enriched with retrieval metadata."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any]
    rank: int                  # 1-indexed rank in the result set

    @property
    def source(self) -> str:
        return self.metadata.get("filename", self.metadata.get("source", "unknown"))

    @property
    def page(self) -> int | None:
        p = self.metadata.get("page")
        return int(p) if p is not None else None

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"RetrievedChunk(rank={self.rank}, score={self.score:.3f}, "
            f"source={self.source!r}, preview={preview!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MMR helper
# ─────────────────────────────────────────────────────────────────────────────


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mmr_rerank(
    query_embedding: list[float],
    candidates: list,           # list[SearchResult]
    candidate_embeddings: list[list[float]],
    top_k: int,
    lambda_: float,
) -> list:
    """
    Maximum Marginal Relevance selection.

    Parameters
    ----------
    query_embedding : list[float]
        Query vector.
    candidates : list[SearchResult]
        Candidate results from initial retrieval.
    candidate_embeddings : list[list[float]]
        Embeddings of the candidate texts (parallel to `candidates`).
    top_k : int
        Number of items to select.
    lambda_ : float
        Trade-off between relevance (1.0) and diversity (0.0).

    Returns
    -------
    list[SearchResult]
        Re-ranked diverse subset of length min(top_k, len(candidates)).
    """
    if not candidates:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    embs = [np.array(e, dtype=np.float32) for e in candidate_embeddings]

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        mmr_scores: list[tuple[float, int]] = []
        for idx in remaining:
            relevance = _cosine_sim(q, embs[idx])
            if selected_indices:
                redundancy = max(_cosine_sim(embs[idx], embs[s]) for s in selected_indices)
            else:
                redundancy = 0.0
            mmr = lambda_ * relevance - (1 - lambda_) * redundancy
            mmr_scores.append((mmr, idx))

        best_idx = max(mmr_scores, key=lambda x: x[0])[1]
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────


class Retriever:
    """
    Retrieves relevant chunks for a query using dense search + optional MMR.

    Usage::

        retriever = Retriever()
        results = retriever.retrieve("What is RAG?")

    Parameters
    ----------
    embedder : SentenceTransformerEmbedder | None
        Override the default embedder singleton.
    store : ChromaVectorStore | None
        Override the default vector store singleton.
    """

    def __init__(self, embedder=None, store=None) -> None:
        from rag.config import get_settings
        from rag.embedding.embedder import get_embedder
        from rag.vectorstore.store import get_vectorstore

        self._cfg = get_settings().retrieval
        self._embedder = embedder or get_embedder()
        self._store = store or get_vectorstore()

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_mmr: bool | None = None,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Parameters
        ----------
        query : str
            The user's question or search query.
        top_k : int | None
            Override default top_k from settings.
        use_mmr : bool | None
            Override default MMR setting.
        metadata_filter : dict | None
            ChromaDB metadata filter (e.g. {"doc_type": "pdf"}).

        Returns
        -------
        list[RetrievedChunk]
            Ranked list of relevant chunks.
        """
        if not query.strip():
            return []

        k = top_k or self._cfg.top_k
        do_mmr = use_mmr if use_mmr is not None else self._cfg.use_mmr
        fetch_k = self._cfg.mmr_fetch_k if do_mmr else k

        # 1. Embed the query
        logger.debug("Embedding query: %r", query[:80])
        query_embedding = self._embedder.embed_query(query)

        # 2. Dense retrieval — fetch more candidates for MMR
        raw_results = self._store.similarity_search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            where=metadata_filter,
            score_threshold=self._cfg.score_threshold,
        )

        if not raw_results:
            logger.warning("No results found for query: %r", query[:80])
            return []

        logger.debug("Retrieved %d candidate(s) from vector store", len(raw_results))

        # 3. Optional MMR re-ranking
        if do_mmr and len(raw_results) > k:
            logger.debug("Applying MMR (lambda=%.2f, k=%d)", self._cfg.mmr_lambda, k)
            # Fetch embeddings for MMR (re-embed the retrieved texts)
            candidate_texts = [r.text for r in raw_results]
            candidate_embeddings = self._embedder.embed_chunks(candidate_texts)
            raw_results = _mmr_rerank(
                query_embedding=query_embedding,
                candidates=raw_results,
                candidate_embeddings=candidate_embeddings,
                top_k=k,
                lambda_=self._cfg.mmr_lambda,
            )
        else:
            raw_results = raw_results[:k]

        # 4. Wrap in RetrievedChunk with ranks
        retrieved: list[RetrievedChunk] = []
        for rank, result in enumerate(raw_results, start=1):
            retrieved.append(
                RetrievedChunk(
                    id=result.id,
                    text=result.text,
                    score=result.score,
                    metadata=result.metadata,
                    rank=rank,
                )
            )

        logger.info(
            "Query retrieved %d chunk(s) | top score: %.3f",
            len(retrieved),
            retrieved[0].score if retrieved else 0.0,
        )
        return retrieved

    def format_context(self, chunks: list[RetrievedChunk], max_chars: int = 6000) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.

        Parameters
        ----------
        chunks : list[RetrievedChunk]
            Retrieved chunks to format.
        max_chars : int
            Hard limit on total context length.

        Returns
        -------
        str
            Formatted context block.
        """
        parts: list[str] = []
        total = 0

        for chunk in chunks:
            source_label = chunk.source
            if chunk.page:
                source_label += f" (page {chunk.page})"
            header = f"[Source {chunk.rank}: {source_label}]"
            body = chunk.text.strip()
            block = f"{header}\n{body}\n"

            if total + len(block) > max_chars:
                # Truncate to fit
                remaining = max_chars - total
                if remaining > 100:
                    block = block[:remaining] + "…"
                    parts.append(block)
                break

            parts.append(block)
            total += len(block)

        return "\n---\n".join(parts)
