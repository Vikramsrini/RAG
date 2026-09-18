"""src/rag/ingestion/__init__.py"""
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.loaders import Document

__all__ = ["IngestionPipeline", "Document"]
