"""
src/rag/config.py
─────────────────
Pydantic v2 settings that load config/settings.yaml and provide fully-typed,
validated configuration objects to every layer of the RAG pipeline.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models — one per config section
# ─────────────────────────────────────────────────────────────────────────────


class PathsConfig(BaseModel):
    documents_dir: Path = Path("data/documents")
    vectorstore_dir: Path = Path("vectorstore")
    logs_dir: Path = Path("logs")


class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = Field(64, ge=1)
    normalize_embeddings: bool = True
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(512, ge=1)
    chunk_overlap: int = Field(64, ge=0)
    min_chunk_size: int = Field(50, ge=1)
    tokenizer_encoding: str = "cl100k_base"
    separators: list[str] = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_chunk_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


class VectorstoreConfig(BaseModel):
    collection_name: str = "rag_documents"
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine"


class RetrievalConfig(BaseModel):
    top_k: int = Field(5, ge=1)
    use_mmr: bool = True
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0)
    mmr_fetch_k: int = Field(20, ge=1)
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1)
    stream: bool = True


class IngestionConfig(BaseModel):
    supported_extensions: list[str] = [".pdf", ".txt", ".md", ".docx", ".html", ".htm", ".csv"]
    recursive: bool = True
    skip_existing: bool = True


class CleaningConfig(BaseModel):
    normalize_unicode: bool = True
    fix_ligatures: bool = True
    collapse_whitespace: bool = True
    remove_control_chars: bool = True
    min_text_length: int = Field(20, ge=0)


# ─────────────────────────────────────────────────────────────────────────────
# Root settings object
# ─────────────────────────────────────────────────────────────────────────────


class Settings(BaseModel):
    paths: PathsConfig = PathsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    vectorstore: VectorstoreConfig = VectorstoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    ingestion: IngestionConfig = IngestionConfig()
    cleaning: CleaningConfig = CleaningConfig()

    # Resolve all paths relative to the project root
    def resolve_paths(self, root: Path) -> "Settings":
        self.paths.documents_dir = root / self.paths.documents_dir
        self.paths.vectorstore_dir = root / self.paths.vectorstore_dir
        self.paths.logs_dir = root / self.paths.logs_dir
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Loader — cached singleton
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_FILE_ENV = "RAG_CONFIG"
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings from YAML. Returns a cached singleton."""
    config_path = Path(os.environ.get(_CONFIG_FILE_ENV, _DEFAULT_CONFIG_PATH))

    if not config_path.exists():
        # Fall back to defaults if no config file found
        settings = Settings()
    else:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        settings = Settings.model_validate(raw)

    # Determine project root (3 levels up from src/rag/config.py)
    project_root = Path(__file__).parent.parent.parent
    settings.resolve_paths(project_root)

    # Ensure required directories exist
    settings.paths.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)

    return settings
