"""
src/rag/embedding/embedder.py
──────────────────────────────
HuggingFace Sentence-Transformer embedder.

Features:
  • Lazy model loading (model only loaded on first call)
  • Automatic device selection: MPS (Apple Silicon) > CUDA > CPU
  • Batched encoding for large document sets
  • Optional L2 normalisation (required for cosine similarity via dot product)
  • Returns numpy arrays — compatible with ChromaDB
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Device auto-detection
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_device(preference: str) -> str:
    """
    Resolve the embedding device.

    Preference order:
      "auto"  →  MPS (Apple Silicon)  >  CUDA  >  CPU
      "mps"   →  MPS if available, else CPU
      "cuda"  →  CUDA if available, else CPU
      "cpu"   →  always CPU
    """
    if preference == "cpu":
        return "cpu"

    try:
        import torch

        if preference in ("auto", "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon) for embeddings")
            return "mps"

        if preference in ("auto", "cuda") and torch.cuda.is_available():
            logger.info("Using CUDA for embeddings")
            return "cuda"
    except ImportError:
        pass

    logger.info("Using CPU for embeddings")
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────


class SentenceTransformerEmbedder:
    """
    Wraps a sentence-transformers model with lazy loading and batching.

    Usage::

        embedder = SentenceTransformerEmbedder()
        vectors = embedder.embed(["Hello world", "RAG is great"])
        # → np.ndarray shape (2, embedding_dim)

    Parameters
    ----------
    model_name : str
        Any HuggingFace sentence-transformers model name.
    batch_size : int
        Number of texts encoded per forward pass.
    normalize : bool
        L2-normalize output vectors (required for cosine similarity).
    device : str
        "auto" | "cpu" | "cuda" | "mps"
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._device = _resolve_device(device)
        self._model = None  # lazy-loaded

    # ── Lazy model loading ────────────────────────────────────────────────────

    @property
    def model(self):
        """Load the model on first access (lazy initialization)."""
        if self._model is None:
            logger.info("Loading embedding model '%s' on %s …", self.model_name, self._device)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)
            logger.info(
                "Model loaded — embedding dim: %d", self.embedding_dim
            )
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        # Load model to get dim (unavoidable on first call)
        _ = self.model
        return self._model.get_sentence_embedding_dimension()

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, texts: Union[str, list[str]]) -> np.ndarray:
        """
        Embed one or more texts.

        Parameters
        ----------
        texts : str | list[str]
            A single string or a list of strings.

        Returns
        -------
        np.ndarray
            Shape (N, embedding_dim) for a list, or (embedding_dim,) for a single string.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Filter out empty strings
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        indices, valid_texts = zip(*non_empty)

        logger.debug("Embedding %d text(s) in batches of %d", len(valid_texts), self.batch_size)

        vectors = self.model.encode(
            list(valid_texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(valid_texts) > self.batch_size,
            convert_to_numpy=True,
        )

        # Re-insert zero vectors for empty strings
        result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for out_idx, vec in zip(indices, vectors):
            result[out_idx] = vec

        return result[0] if single else result

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string and return a Python list (for ChromaDB)."""
        vec = self.embed(query)
        return vec.tolist()

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple chunk texts and return as list of lists (for ChromaDB)."""
        vectors = self.embed(texts)
        return vectors.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformerEmbedder:
    """Return a cached embedder configured from global Settings."""
    from rag.config import get_settings

    cfg = get_settings().embedding
    return SentenceTransformerEmbedder(
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        normalize=cfg.normalize_embeddings,
        device=cfg.device,
    )
