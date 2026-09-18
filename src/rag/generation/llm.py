"""
src/rag/generation/llm.py
──────────────────────────
Ollama LLM client with RAG-optimised prompt templating.

Features:
  • Streaming generation (token-by-token, perfect for Streamlit)
  • Non-streaming for simple programmatic use
  • Configurable system prompt and RAG template
  • Graceful error handling (connection errors → clear user message)
  • Model availability check on first use
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Generator, Iterator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise, helpful AI assistant powered by a Retrieval-Augmented Generation (RAG) system.

Your task is to answer the user's question based ONLY on the provided context passages.

Rules:
1. Answer using only information from the provided context. Do not rely on prior knowledge.
2. If the context does not contain enough information to answer the question, say:
   "I couldn't find a clear answer in the provided documents."
3. Be concise and factual. Cite source numbers (e.g., [Source 1]) when referencing specific passages.
4. Never fabricate information.
"""

_RAG_PROMPT_TEMPLATE = """Here are the relevant passages retrieved from the knowledge base:

{context}

---

User question: {question}

Please provide a thorough, accurate answer based on the above context:"""


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """A complete (non-streamed) response from the LLM."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ─────────────────────────────────────────────────────────────────────────────
# OllamaLLM
# ─────────────────────────────────────────────────────────────────────────────


class OllamaLLM:
    """
    Ollama LLM client.

    Supports both streaming and non-streaming generation.

    Usage::

        llm = OllamaLLM()
        # Non-streaming
        response = llm.generate(context="...", question="What is RAG?")
        print(response.text)

        # Streaming
        for token in llm.stream(context="...", question="What is RAG?"):
            print(token, end="", flush=True)

    Parameters
    ----------
    model : str
        Ollama model name (must be pulled: `ollama pull <model>`).
    base_url : str
        Ollama server URL (default: http://localhost:11434).
    temperature : float
        Generation temperature (0 = deterministic, 1 = creative).
    max_tokens : int
        Maximum tokens to generate.
    system_prompt : str | None
        Override the default system prompt.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or _SYSTEM_PROMPT
        self._client = None
        self._checked = False

    # ── Lazy client ───────────────────────────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError as e:
                raise ImportError("Install ollama: pip install ollama") from e
        return self._client

    def check_availability(self) -> bool:
        """Verify Ollama is running and the model is available."""
        if self._checked:
            return True
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            # Normalize: strip ":latest" suffix for comparison
            normalized = [m.split(":")[0] for m in model_names]
            target = self.model.split(":")[0]
            if target not in normalized:
                logger.warning(
                    "Model '%s' not found locally. Available: %s\n"
                    "Run: ollama pull %s",
                    self.model,
                    model_names,
                    self.model,
                )
                return False
            self._checked = True
            logger.info("Ollama is running. Model '%s' is available.", self.model)
            return True
        except Exception as exc:
            logger.error("Cannot connect to Ollama at %s: %s", self.base_url, exc)
            return False

    # ── Prompt building ───────────────────────────────────────────────────────

    def _build_messages(self, context: str, question: str) -> list[dict]:
        """Build the chat message list for the Ollama API."""
        user_content = _RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, context: str, question: str) -> LLMResponse:
        """
        Generate a complete (non-streamed) response.

        Parameters
        ----------
        context : str
            Formatted retrieved context (from Retriever.format_context).
        question : str
            The user's question.

        Returns
        -------
        LLMResponse
            Complete response with token usage stats.
        """
        messages = self._build_messages(context, question)
        logger.debug("Generating response for: %r", question[:80])

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                stream=False,
            )
            text = response.message.content
            usage = response.usage if hasattr(response, "usage") else None
            return LLMResponse(
                text=text,
                model=self.model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc, exc_info=True)
            return LLMResponse(
                text=f"⚠️ Generation failed: {exc}\n\nPlease ensure Ollama is running with: `ollama serve`",
                model=self.model,
            )

    def stream(self, context: str, question: str) -> Iterator[str]:
        """
        Stream tokens one by one.

        Yields
        ------
        str
            Individual token strings as they are generated.
        """
        messages = self._build_messages(context, question)
        logger.debug("Streaming response for: %r", question[:80])

        try:
            for chunk in self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                stream=True,
            ):
                token = chunk.message.content
                if token:
                    yield token
        except Exception as exc:
            logger.error("Streaming failed: %s", exc)
            yield f"\n\n⚠️ Streaming error: {exc}\n\nPlease ensure Ollama is running with: `ollama serve`"

    def list_local_models(self) -> list[str]:
        """Return names of all locally available Ollama models."""
        try:
            return [m.model for m in self.client.list().models]
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_llm() -> OllamaLLM:
    """Return a cached OllamaLLM configured from global Settings."""
    from rag.config import get_settings

    cfg = get_settings().llm
    return OllamaLLM(
        model=cfg.model,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
