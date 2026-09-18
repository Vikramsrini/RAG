"""
src/rag/ingestion/cleaner.py
─────────────────────────────
Text cleaning & normalization pipeline.

Stages (applied in order):
  1. Unicode NFC normalization
  2. Ligature expansion  (ﬁ → fi, ﬂ → fl, etc.)
  3. Control character removal (non-printable ASCII)
  4. Whitespace collapsing (tabs → space, multiple spaces → one)
  5. Blank line collapsing (more than 2 consecutive → 2)
  6. Leading/trailing whitespace strip per line
  7. Length filter (discard very short outputs)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Ligature map — covers common PDF extraction artefacts
# ─────────────────────────────────────────────────────────────────────────────

_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "st",   # ﬅ
    "\ufb06": "st",   # ﬆ
    "\u00e6": "ae",   # æ  (keep if not desired)
    "\u0153": "oe",   # œ
    "\u00df": "ss",   # ß
}

_LIGATURE_TABLE = str.maketrans(_LIGATURE_MAP)

# Regex patterns compiled once at import time
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_BULLET_CHARS_RE = re.compile(r"^[\u2022\u2023\u25e6\u2043\u2219•·▪▸►→–—]\s*", re.MULTILINE)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_HEADER_FOOTER_RE = re.compile(
    r"(?:page\s+\d+\s+of\s+\d+|confidential|all rights reserved|©.*?\d{4})",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# TextCleaner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CleanerConfig:
    normalize_unicode: bool = True
    fix_ligatures: bool = True
    remove_control_chars: bool = True
    collapse_whitespace: bool = True
    strip_bullet_chars: bool = True
    remove_isolated_page_numbers: bool = True
    remove_common_boilerplate: bool = True
    min_text_length: int = 20


class TextCleaner:
    """
    Stateless text-cleaning pipeline.

    Usage::

        cleaner = TextCleaner(config)
        clean = cleaner.clean(raw_text)
    """

    def __init__(self, config: CleanerConfig | None = None) -> None:
        self.config = config or CleanerConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        Apply all enabled cleaning stages and return the cleaned text.
        Returns an empty string if the result is shorter than min_text_length.
        """
        if not text:
            return ""

        cfg = self.config

        if cfg.normalize_unicode:
            text = self._normalize_unicode(text)

        if cfg.fix_ligatures:
            text = self._fix_ligatures(text)

        if cfg.remove_control_chars:
            text = self._remove_control_chars(text)

        if cfg.strip_bullet_chars:
            text = _BULLET_CHARS_RE.sub("", text)

        if cfg.remove_isolated_page_numbers:
            text = _PAGE_NUMBER_RE.sub("", text)

        if cfg.remove_common_boilerplate:
            text = _HEADER_FOOTER_RE.sub("", text)

        if cfg.collapse_whitespace:
            text = self._collapse_whitespace(text)

        text = text.strip()

        if len(text) < cfg.min_text_length:
            logger.debug("Discarding short text (%d chars): %r", len(text), text[:40])
            return ""

        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean a batch of texts, filtering out empty results."""
        cleaned = [self.clean(t) for t in texts]
        return [t for t in cleaned if t]

    # ── Private stages ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """NFC normalization — combines composed characters."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _fix_ligatures(text: str) -> str:
        """Replace typographic ligatures with ASCII equivalents."""
        return text.translate(_LIGATURE_TABLE)

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """Strip non-printable ASCII control characters (keep \\n, \\r, \\t)."""
        return _CONTROL_CHARS_RE.sub("", text)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """
        1. Tabs → single space within lines.
        2. Multiple spaces → single space.
        3. Three or more consecutive newlines → two newlines.
        4. Strip trailing whitespace on each line.
        """
        # Collapse horizontal whitespace
        text = _MULTI_SPACE_RE.sub(" ", text)
        # Strip trailing spaces per line
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines)
        # Collapse excess blank lines
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Convenience singleton factory
# ─────────────────────────────────────────────────────────────────────────────


def build_cleaner_from_settings() -> TextCleaner:
    """Build a TextCleaner from the global Settings object."""
    from rag.config import get_settings

    cfg = get_settings().cleaning
    return TextCleaner(
        CleanerConfig(
            normalize_unicode=cfg.normalize_unicode,
            fix_ligatures=cfg.fix_ligatures,
            remove_control_chars=cfg.remove_control_chars,
            collapse_whitespace=cfg.collapse_whitespace,
            min_text_length=cfg.min_text_length,
        )
    )
