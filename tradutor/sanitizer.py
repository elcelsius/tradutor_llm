"""
Sanitizacao agressiva contra alucinacoes e ruidos dos modelos.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple


META_PATTERNS = [
    r"parece que voc[eˆ] est[ a]",
    r"como um modelo de linguagem",
    r"n[aÆ]o posso",
    r"n[aÆ]o sou capaz",
    r"desculp",
    r"nÆo posso ajudar",
    r"eu sou apenas",
    r"como um assistente",
    r"as an ai language model",
    r"i am an ai",
    r"i cannot provide",
    r"i'm just an ai",
    r"as a language model",
    r"^\s*mudanc(a|‡)as e justificativas[:]?.*$",
    r"^\s*alterac(ao|Æo|oes|äes) realizadas[:]?.*$",
    r"^\s*(nesta|nessa) revis(ao|Æo).*$",
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
    leading_noise_removed: bool = False
    removed_lines_count: int = 0
    collapsed_repetitions: int = 0


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
    Remove sequencias longas repetidas (loop detectavel).
    Considera repeticoes consecutivas de blocos >= 50 caracteres.
    """
    pattern = re.compile(r"(.{50,}?)(?:\s+\1){1,}", flags=re.DOTALL)
    new_text, count = pattern.subn(lambda m: m.group(1), text)
    return new_text, count


def remove_leading_noise(text: str) -> str:
    """
    Remove ruido obvio no topo (pontuacao solta, OCR quebrado) e preserva
    qualquer linha com letras/digitos ou terminacao de frase.
    """
    lines = text.splitlines()
    cleaned: List[str] = []
    started = False

    for line in lines:
        stripped = line.strip()

        if not started:
            if not stripped:
                continue
            if (
                len(stripped) <= 12
                and not re.search(r"[A-Za-zÀ-ÿ0-9]", stripped)
                and not re.search(r"[.!?…]$", stripped)
            ):
                continue

            started = True

        cleaned.append(line)

    return "\n".join(cleaned)


def sanitize_text(
    text: str,
    logger: logging.Logger | None = None,
    fail_on_contamination: bool = True,
    collapse_repeated_lines: bool = True,
    collapse_repeated_paragraphs: bool = True,
    remove_repeated_sequences: bool = True,
    strip_empty_lines: bool = True,
    apply_leading_noise_filter: bool = True,
) -> Tuple[str, SanitizationReport]:
    """
    Sanitiza saidas de LLM para reduzir alucinacao/ruido.

    Retorna texto limpo e um relatorio da sanitizacao.
    Levanta ValueError se o resultado ficar vazio.
    """
    report = SanitizationReport()

    text, count = _remove_think_blocks(text)
    report.removed_think_blocks = count

    text, meta_removed, contamination = _remove_meta_lines(text)
    report.removed_meta_lines = meta_removed
    report.contamination_detected = contamination

    repeated_lines = 0
    if collapse_repeated_lines:
        text, repeated_lines = _collapse_repeated_lines(text)
    report.removed_repeated_lines = repeated_lines

    seq_removed = 0
    repeated_paragraphs = 0
    if remove_repeated_sequences:
        text, seq_removed = _remove_repeated_sequences(text)
    if collapse_repeated_paragraphs:
        text, repeated_paragraphs = _collapse_repeated_paragraphs(text)
    report.removed_repeated_paragraphs = seq_removed + repeated_paragraphs

    empty = 0
    if strip_empty_lines:
        text, empty = _strip_empty_lines(text)
    report.removed_empty_lines = empty
    report.collapsed_repetitions = seq_removed + repeated_paragraphs

    before_noise = text
    if apply_leading_noise_filter:
        text = remove_leading_noise(text)
    report.leading_noise_removed = apply_leading_noise_filter and text != before_noise
    text = text.replace("<think>", "").replace("</think>", "")
    report.removed_lines_count = report.removed_meta_lines + report.removed_repeated_lines + report.removed_empty_lines

    text = text.strip()
    if not text:
        if logger:
            logger.error("Sanitizacao resultou em texto vazio.")
        raise ValueError("Texto vazio apos sanitizacao.")

    if fail_on_contamination and report.contamination_detected:
        raise ValueError("Contaminacao detectada na saida do modelo.")

    return text, report


def sanitize_translation_output(
    text: str,
    logger: logging.Logger | None = None,
    fail_on_contamination: bool = False,
) -> Tuple[str, SanitizationReport]:
    """
    Versao mais leve para traducao EN->PT:
    - remove <think> e metacomentarios
    - preserva linhas/paragrafos repetidos e espacos em branco
    - aplica filtro de ruido inicial apenas para lixo evidente
    """
    return sanitize_text(
        text,
        logger=logger,
        fail_on_contamination=fail_on_contamination,
        collapse_repeated_lines=False,
        collapse_repeated_paragraphs=False,
        remove_repeated_sequences=False,
        strip_empty_lines=False,
        apply_leading_noise_filter=True,
    )


def sanitize_refine_output(text: str) -> str:
    """
    Sanitizacao leve para saida do refinador:
    - remove tags <think> e </think>
    - remove cabecalhos "Texto refinado:" / "Refined text:"
    - remove marcadores de traducao remanescentes (### TEXTO_TRADUZIDO_*)
    - remove blocos de glossario legado (===GLOSSARIO_SUGERIDO_INICIO=== ... FIM=== ou orfaos)
    - remove espacos extras nas extremidades
    Nao aplica regras agressivas nem corta paragrafos.
    """
    cleaned = text.replace("<think>", "").replace("</think>", "")
    filtered_lines = []
    for line in cleaned.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("texto refinado:") or lowered.startswith("refined text:"):
            continue
        filtered_lines.append(line)
    cleaned = "\n".join(filtered_lines)

    cleaned = re.sub(
        r"### TEXTO_TRADUZIDO_INICIO.*?### TEXTO_TRADUZIDO_FIM",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"### TEXTO_TRADUZIDO_[A-Z_]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"===GLOSSARIO_SUGERIDO_INICIO===.*?===GLOSSARIO_SUGERIDO_FIM===",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    start = cleaned.find("===GLOSSARIO_SUGERIDO_INICIO===")
    end = cleaned.find("===GLOSSARIO_SUGERIDO_FIM===")
    if start != -1 and (end == -1 or end < start):
        pre = cleaned[:start]
        pre_rstrip = pre.rstrip()
        if pre_rstrip.endswith('"""'):
            pre = pre_rstrip[:-3]
        cleaned = pre.rstrip()

    return cleaned.strip()


def log_report(report: SanitizationReport, logger: logging.Logger, prefix: str) -> None:
    """Registra o relatorio de sanitizacao com prefixo."""
    logger.debug(
        "%s sanitizacao -> think:%d meta:%d rep_linhas:%d rep_parag:%d vazias:%d contam:%s leading_noise:%s colapsos:%d",
        prefix,
        report.removed_think_blocks,
        report.removed_meta_lines,
        report.removed_repeated_lines,
        report.removed_repeated_paragraphs,
        report.removed_empty_lines,
        report.contamination_detected,
        report.leading_noise_removed,
        report.collapsed_repetitions,
    )
