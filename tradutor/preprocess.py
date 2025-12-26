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
    r"zerobooks?",
    r"jnovels?",
    r"newsletter",
    r"discord\.gg",
    r"stay up to date",
    r"download(?:ing)? our mobile app",
]

NOISE_PARAGRAPH_PATTERNS: Final[list[str]] = [
    r"stay up to date",
    r"download(?:ing)? our mobile app",
    r"zerobooks",
    r"jnovels",
    r"join our discord",
    r"newsletter",
    r"follow us",
    r"support us",
    r"read (more|the latest) on",
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


def remove_noise_blocks(text: str) -> str:
    """Remove paragrafos que pareçam ser ads/newsletter/discord etc."""
    paragraphs = text.split("\n\n")
    cleaned: list[str] = []
    for para in paragraphs:
        norm = para.lower().strip()
        if not norm:
            cleaned.append("")
            continue
        if any(re.search(pat, norm, flags=re.IGNORECASE) for pat in NOISE_PARAGRAPH_PATTERNS):
            continue
        cleaned.append(para.strip())
    # reintroduz quebras duplas
    return "\n\n".join(p for p in cleaned if p != "")


def strip_front_matter(text: str) -> str:
    """
    Remove tudo antes de Prologue/Chapter 1.
    """
    lines = text.splitlines()
    start_idx = None
    marker_re = re.compile(r"^(prologue|chapter\s*1:?|cap[ií]tulo\s*1:?|ep[ií]logo)", re.IGNORECASE)
    for idx, ln in enumerate(lines):
        if marker_re.match(ln.strip()):
            start_idx = idx
            break
    if start_idx is None:
        return text
    return "\n".join(lines[start_idx:]).lstrip()


def preprocess_text(raw_text: str, logger: Optional[logging.Logger] = None, *, skip_front_matter: bool = False) -> str:
    """
    Pré-processa o texto bruto extraído do PDF:
    - Normaliza quebras de linha
    - Remove rodapés/watermarks
    - Mantém todo o conteúdo original
    """

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    if skip_front_matter:
        text = strip_front_matter(text)

    for pattern in FOOTER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()
    text = remove_noise_blocks(text)

    if logger is not None:
        logger.debug("Texto pré-processado: %d caracteres", len(text))

    return text


def paragraphs_from_text(clean_text: str) -> List[str]:
    """Divide texto limpo em parágrafos usando quebras duplas."""
    return [p.strip() for p in clean_text.split("\n\n") if p.strip()]


def chunk_for_translation(paragraphs: List[str], max_chars: int, logger: logging.Logger) -> List[str]:
    """
    Chunk seguro para tradução com ajuste leve por fronteira de frase.

    Usa max_chars como alvo, mas permite pequeno lookahead para fechar
    o chunk no fim de frase (., ?, !) evitando cortar falas.
    """
    text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    if not text:
        return []

    boundary_re = re.compile(r"\n\n|[.!?](?:['\"”])?(?=\s|\n|$)")
    chunks: List[str] = []
    start = 0
    total_len = len(text)
    lookahead = 400  # permite estouro controlado para terminar frase
    consumed = 0

    while start < total_len:
        target_end = start + max_chars
        hard_end = min(total_len, start + max_chars + lookahead)

        if target_end >= total_len:
            last_slice = text[start:]
            chunks.append(last_slice.strip())
            consumed += len(last_slice)
            break

        window = text[start:hard_end]
        after_target: int | None = None
        before_target: int | None = None

        for match in boundary_re.finditer(window):
            end_pos = start + match.end()
            if target_end <= end_pos <= hard_end:
                after_target = end_pos
            elif end_pos < target_end:
                before_target = end_pos

        if after_target:
            chunk_end = after_target
            logger.debug(
                "tradução: chunk fechado em fim de frase após lookahead (len=%d)",
                chunk_end - start,
            )
        elif before_target:
            chunk_end = before_target
            logger.debug(
                "tradução: chunk fechado em limite seguro antes do alvo (len=%d)",
                chunk_end - start,
            )
        else:
            chunk_end = min(target_end, total_len)
            logger.debug("tradução: chunk fechado no alvo (len=%d)", chunk_end - start)

        if chunk_end <= start:
            chunk_end = min(target_end, total_len)

        raw_slice = text[start:chunk_end]
        chunks.append(raw_slice.strip())
        consumed += len(raw_slice)
        start = chunk_end

    if consumed != total_len:
        logger.warning("tradução: soma dos chunks (%d) difere do texto original (%d)", consumed, total_len)

    return chunks


def chunk_for_refine(paragraphs: List[str], max_chars: int, logger: logging.Logger) -> List[str]:
    """Chunk seguro para refine com limite estrito."""
    return chunk_by_paragraphs(paragraphs, max_chars=max_chars, logger=logger, label="refine")
