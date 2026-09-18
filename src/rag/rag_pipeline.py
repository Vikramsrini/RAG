"""
src/rag/rag_pipeline.py
────────────────────────
Top-level RAG orchestrator.

Ties together:
  IngestionPipeline → Embedder → ChromaVectorStore → Retriever → OllamaLLM

Usage::

    pipeline = RAGPipeline()

    # Ingest documents
    pipeline.ingest(Path("data/documents"))

    # Query
    response = pipeline.query("What is the main topic of the document?")
    print(response.answer)
    for chunk in response.sources:
        print(f"  [{chunk.rank}] {chunk.source} — score: {chunk.score:.3f}")

    # Streaming
    for token in pipeline.stream("Summarize the key findings"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RAGResponse:
    """Complete RAG response with answer and provenance."""

    question: str
    answer: str
    sources: list = field(default_factory=list)     # list[RetrievedChunk]
    context_used: str = ""

    @property
    def num_sources(self) -> int:
        return len(self.sources)

    def __str__(self) -> str:
        lines = [f"Q: {self.question}", f"A: {self.answer}", ""]
        for src in self.sources:
            lines.append(f"  [{src.rank}] {src.source} (score: {src.score:.3f})")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# RAGPipeline
# ─────────────────────────────────────────────────────────────────────────────


class RAGPipeline:
    """
    End-to-end RAG pipeline — ingest, embed, store, retrieve, generate.

    Parameters
    ----------
    show_progress : bool
        Show Rich progress bars during ingestion.
    """

    def __init__(self, show_progress: bool = True) -> None:
        from rag.config import get_settings
        from rag.embedding.embedder import get_embedder
        from rag.generation.llm import get_llm
        from rag.ingestion.pipeline import IngestionPipeline
        from rag.retrieval.retriever import Retriever
        from rag.vectorstore.store import get_vectorstore

        self._settings = get_settings()
        self._ingestion = IngestionPipeline(show_progress=show_progress)
        self._embedder = get_embedder()
        self._store = get_vectorstore()
        self._retriever = Retriever(embedder=self._embedder, store=self._store)
        self._llm = get_llm()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, path: Path | str, force: bool = False) -> int:
        """
        Ingest a file or directory into the vector store.

        Parameters
        ----------
        path : Path | str
            File or directory to ingest.
        force : bool
            Re-ingest files even if already cached.

        Returns
        -------
        int
            Number of chunks added to the vector store.
        """
        path = Path(path)
        logger.info("Starting ingestion of: %s", path)

        # 1. Load → Clean → Chunk
        chunks = self._ingestion.run(path, force=force)
        if not chunks:
            logger.warning("No chunks produced from: %s", path)
            return 0

        # 2. Embed all chunks in one batched call
        logger.info("Embedding %d chunks …", len(chunks))
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_chunks(texts)

        # 3. Upsert into ChromaDB
        ids = [c.id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        n_upserted = self._store.upsert_chunks(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info("Ingestion complete: %d chunks stored", n_upserted)
        return n_upserted

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> RAGResponse:
        """
        Answer a question using retrieval-augmented generation.

        Parameters
        ----------
        question : str
            The user's question.
        top_k : int | None
            Override default number of retrieved chunks.
        metadata_filter : dict | None
            Filter retrieved chunks by metadata (e.g. doc type, source).

        Returns
        -------
        RAGResponse
            Complete response with answer and source citations.
        """
        if not question.strip():
            return RAGResponse(
                question=question,
                answer="Please provide a non-empty question.",
            )

        # 1. Retrieve
        chunks = self._retriever.retrieve(
            query=question,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        if not chunks:
            return RAGResponse(
                question=question,
                answer=(
                    "I couldn't find any relevant information in the knowledge base. "
                    "Please ensure documents have been ingested first."
                ),
                sources=[],
            )

        # 2. Format context
        context = self._retriever.format_context(chunks)

        # 3. Generate
        logger.debug("Generating answer for: %r", question[:80])
        response = self._llm.generate(context=context, question=question)

        return RAGResponse(
            question=question,
            answer=response.text,
            sources=chunks,
            context_used=context,
        )

    def stream(
        self,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> tuple[list, Iterator[str]]:
        """
        Stream an answer token-by-token.

        Returns a tuple of (sources, token_iterator) so the caller
        can display sources before/during streaming.

        Usage::

            sources, tokens = pipeline.stream("What is RAG?")
            for token in tokens:
                print(token, end="", flush=True)
        """
        if not question.strip():
            return [], iter(["Please provide a non-empty question."])

        # 1. Retrieve
        chunks = self._retriever.retrieve(
            query=question,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        if not chunks:
            def _no_context():
                yield (
                    "I couldn't find any relevant information in the knowledge base. "
                    "Please ensure documents have been ingested first."
                )
            return [], _no_context()

        # 2. Format context
        context = self._retriever.format_context(chunks)

        # 3. Stream
        return chunks, self._llm.stream(context=context, question=question)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return statistics about the current knowledge base."""
        return self._store.stats()

    def clear_knowledge_base(self) -> None:
        """Delete all ingested documents from the vector store."""
        self._store.clear()
        logger.warning("Knowledge base cleared.")
