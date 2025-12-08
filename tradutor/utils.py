"""
Funções utilitárias compartilhadas.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura logging simples para console."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("tradutor")


def ensure_dir(path: Path) -> None:
    """Cria diretório se não existir."""
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path, encoding: str = "utf-8") -> str:
    """Lê arquivo texto com encoding definido."""
    return path.read_text(encoding=encoding)


def write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Escreve texto garantindo diretório."""
    ensure_dir(path.parent)
    path.write_text(content, encoding=encoding)


def chunk_by_paragraphs(
    paragraphs: Sequence[str],
    max_chars: int,
    logger: logging.Logger,
    label: str,
) -> List[str]:
    """
    Agrupa parágrafos até `max_chars`, evitando chunks gigantes.

    Se um parágrafo exceder max_chars, faz corte duro dentro dele para manter o limite.
    """
    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)
        if para_len > max_chars:
            logger.warning("%s: parágrafo > max_chars (%d). Corta duro.", label, para_len)
            start = 0
            while start < para_len:
                end = min(start + max_chars, para_len)
                slice_text = para[start:end].strip()
                if slice_text:
                    chunks.append(slice_text)
                start = end
            continue

        if current_len + para_len + 2 <= max_chars:
            buffer.append(para)
            current_len += para_len + 2
        else:
            if buffer:
                chunks.append("\n\n".join(buffer))
            buffer = [para]
            current_len = para_len + 2

    if buffer:
        chunks.append("\n\n".join(buffer))

    # Garantir nenhum chunk vazio e nenhum chunk gigante
    safe_chunks = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if len(c) > max_chars:
            logger.warning("%s: chunk > max_chars pós-montagem (%d). Cortando.", label, len(c))
            start = 0
            while start < len(c):
                end = min(start + max_chars, len(c))
                safe_chunks.append(c[start:end].strip())
                start = end
        else:
            safe_chunks.append(c)
    return safe_chunks


def timed(fn, *args, **kwargs) -> Tuple[float, any]:
    """Executa função e retorna (segundos, resultado)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def dedent_triple(text: str) -> str:
    """Remove indentação mínima preservando quebras."""
    import textwrap

    return textwrap.dedent(text).strip()
