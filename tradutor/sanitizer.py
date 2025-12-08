"""
Sanitização agressiva contra alucinações e ruídos dos modelos.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple


META_PATTERNS = [
    r"parece que voc[eê] est[áa]",
    r"como um modelo de linguagem",
    r"n[aã]o posso",
    r"n[aã]o sou capaz",
    r"desculp",
    r"não posso ajudar",
    r"eu sou apenas",
    r"como um assistente",
    r"as an ai language model",
    r"i am an ai",
    r"i cannot provide",
    r"i'm just an ai",
    r"as a language model",
    r"^\s*mudanc(a|ç)as e justificativas[:]?.*$",
    r"^\s*alterac(ao|ão|oes|ões) realizadas[:]?.*$",
    r"^\s*(nesta|nessa) revis(ao|ão).*$",
    r"^\s*(justificativa|racionalidade|rationale).*$",
    r"^\s*em resumo.*$",
    r"^\s*resumo[: ].*$",
]


@dataclass
class SanitizationReport:
    removed_think_blocks: int = 0
    removed_meta_lines: int = 0
    removed_repeated_lines: int = 0
    removed_repeated_paragraphs: int = 0
    removed_empty_lines: int = 0
    contamination_detected: bool = False


def _remove_think_blocks(text: str) -> Tuple[str, int]:
    pattern = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
    new_text, count = pattern.subn("", text)
    return new_text, count


def _remove_meta_lines(text: str) -> Tuple[str, int, bool]:
    lines = text.splitlines()
    kept: List[str] = []
    removed = 0
    contamination = False
    for line in lines:
        lowered = line.lower()
        if any(re.search(pat, lowered) for pat in META_PATTERNS):
            removed += 1
            contamination = True
            continue
        kept.append(line)
    return "\n".join(kept), removed, contamination


def _collapse_repeated_lines(text: str) -> Tuple[str, int]:
    lines = text.splitlines()
    kept: List[str] = []
    removed = 0
    prev = None
    for line in lines:
        if prev is not None and line.strip() and prev.strip() == line.strip():
            removed += 1
            continue
        kept.append(line)
        prev = line
    return "\n".join(kept), removed


def _collapse_repeated_paragraphs(text: str) -> Tuple[str, int]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    kept: List[str] = []
    removed = 0
    prev = None
    for p in paragraphs:
        if prev is not None and p == prev:
            removed += 1
            continue
        kept.append(p)
        prev = p
    return "\n\n".join(kept), removed


def _strip_empty_lines(text: str) -> Tuple[str, int]:
    lines = text.splitlines()
    kept: List[str] = []
    removed = 0
    for line in lines:
        if line.strip() == "":
            removed += 1
            continue
        kept.append(line.rstrip())
    return "\n".join(kept), removed


def _remove_repeated_sequences(text: str) -> Tuple[str, int]:
    """
    Remove sequências longas repetidas (loop detectável).
    Considera repetições consecutivas de blocos >= 50 caracteres.
    """
    pattern = re.compile(r"(.{50,}?)(?:\s+\1){1,}", flags=re.DOTALL)
    new_text, count = pattern.subn(lambda m: m.group(1), text)
    return new_text, count


def sanitize_text(
    text: str,
    logger: logging.Logger | None = None,
    fail_on_contamination: bool = True,
) -> Tuple[str, SanitizationReport]:
    """
    Sanitiza agressivamente saídas de LLM para reduzir alucinação/ruído.

    Retorna texto limpo e um relatório da sanitização.
    Levanta ValueError se o resultado ficar vazio.
    """
    report = SanitizationReport()

    text, count = _remove_think_blocks(text)
    report.removed_think_blocks = count

    text, meta_removed, contamination = _remove_meta_lines(text)
    report.removed_meta_lines = meta_removed
    report.contamination_detected = contamination

    text, repeated_lines = _collapse_repeated_lines(text)
    report.removed_repeated_lines = repeated_lines

    text, seq_removed = _remove_repeated_sequences(text)
    text, repeated_paragraphs = _collapse_repeated_paragraphs(text)
    report.removed_repeated_paragraphs = seq_removed + repeated_paragraphs

    text, empty = _strip_empty_lines(text)
    report.removed_empty_lines = empty

    text = text.strip()
    if not text:
        if logger:
            logger.error("Sanitização resultou em texto vazio.")
        raise ValueError("Texto vazio após sanitização.")

    if fail_on_contamination and report.contamination_detected:
        raise ValueError("Contaminação detectada na saída do modelo.")

    return text, report


def log_report(report: SanitizationReport, logger: logging.Logger, prefix: str) -> None:
    """Registra o relatório de sanitização com prefixo."""
    logger.debug(
        "%s sanitização -> think:%d meta:%d rep_linhas:%d rep_parag:%d vazias:%d contam:%s",
        prefix,
        report.removed_think_blocks,
        report.removed_meta_lines,
        report.removed_repeated_lines,
        report.removed_repeated_paragraphs,
        report.removed_empty_lines,
        report.contamination_detected,
    )
