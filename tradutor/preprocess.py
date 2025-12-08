"""
Pré-processamento de PDFs: extração de texto, limpeza e chunking seguro.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - fallback para ambientes sem PyMuPDF
    class _DummyDoc:
        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            return False

        def __iter__(self):
            return iter([])

        @property
        def pages(self):
            return []

        def __len__(self):
            return 0

    class _DummyFitz:
        def open(self, *args, **kwargs):
            return _DummyDoc()

    fitz = _DummyFitz()  # type: ignore

from .utils import chunk_by_paragraphs

# Watermarks de sites/grupos de scan.
FOOTER_PATTERNS: Final[list[str]] = [
    r"\bPage\s+\d+\b",
    r"Goldenagato \| mp4directs\.com",
    r"mp4directs\.com",
]


def extract_text_from_pdf(path: Path, logger: logging.Logger) -> str:
    """Extrai texto de um PDF usando PyMuPDF."""
    with fitz.open(path) as doc:
        pages = [page.get_text() or "" for page in doc]
    text = "\n".join(pages)
    logger.debug("PDF %s extraído: %d caracteres", path.name, len(text))
    return text


def _remove_headers_footers(text: str) -> str:
    lines = text.splitlines()
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        # Números de página isolados ou cabeçalhos típicos
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if len(stripped) <= 5 and stripped.isupper():
            continue
        if re.search(r"\bpage\b", stripped, re.IGNORECASE):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def _remove_hyphenation(text: str) -> str:
    return re.sub(r"(\w+)-\s*\n(\w+)", r"\1\2\n", text)


def _join_broken_lines(text: str) -> str:
    lines = text.splitlines()
    joined: List[str] = []
    buffer: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                joined.append(" ".join(buffer))
                buffer = []
            continue
        if re.search(r"[.!?…]$", stripped):
            buffer.append(stripped)
            joined.append(" ".join(buffer))
            buffer = []
        else:
            buffer.append(stripped)

    if buffer:
        joined.append(" ".join(buffer))

    return "\n\n".join(joined)


def preprocess_text(raw_text: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Pré-processa o texto bruto extraído do PDF:
    - Normaliza quebras de linha
    - Remove rodapés/watermarks
    - Mantém todo o conteúdo original
    """

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    for pattern in FOOTER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    if logger is not None:
        logger.debug("Texto pré-processado: %d caracteres", len(text))

    return text


def paragraphs_from_text(clean_text: str) -> List[str]:
    """Divide texto limpo em parágrafos usando quebras duplas."""
    return [p.strip() for p in clean_text.split("\n\n") if p.strip()]


def chunk_for_translation(paragraphs: List[str], max_chars: int, logger: logging.Logger) -> List[str]:
    """Chunk seguro para tradução com limite estrito."""
    return chunk_by_paragraphs(paragraphs, max_chars=max_chars, logger=logger, label="tradução")


def chunk_for_refine(paragraphs: List[str], max_chars: int, logger: logging.Logger) -> List[str]:
    """Chunk seguro para refine com limite estrito."""
    return chunk_by_paragraphs(paragraphs, max_chars=max_chars, logger=logger, label="refine")
