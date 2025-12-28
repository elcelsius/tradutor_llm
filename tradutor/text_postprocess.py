from __future__ import annotations

import re
from typing import Tuple

from .quote_fix import _collapse_blank_lines_in_quotes

TRIPLE_QUOTES_EOF_RE = re.compile(r'"""+\s*$')
DIALOGUE_BREAK_PATTERNS = (
    (re.compile(r"”{2,}\s*“"), "”\n\n“"),
    (re.compile(r'"{2,}\s*"'), '"\n\n"'),
    (re.compile(r"”\s*“"), "”\n\n“"),
    (re.compile(r'"\s*"'), '"\n\n"'),
)


def _remove_trailing_triple_quotes(text: str) -> Tuple[str, int]:
    """Remove aspas triplas soltas no fim de linha."""
    total = 0
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        newline = ""
        if line.endswith("\r\n"):
            newline = "\r\n"
            body = line[:-2]
        elif line.endswith("\n"):
            newline = "\n"
            body = line[:-1]
        elif line.endswith("\r"):
            newline = "\r"
            body = line[:-1]
        else:
            body = line

        cleaned, count = TRIPLE_QUOTES_EOF_RE.subn("", body)
        total += count
        lines.append(cleaned + newline)
    return "".join(lines), total


def _split_glued_dialogues(text: str) -> Tuple[str, int]:
    """Separa falas coladas (com ou sem espaço)."""
    cleaned, stats = normalize_dialogue_breaks(text)
    return cleaned, stats.get("dialogue_splits", 0)


def normalize_dialogue_breaks(text: str) -> tuple[str, dict]:
    """
    Insere quebras de paragrafo entre falas consecutivas coladas.
    Nao altera o conteudo alem das quebras. Retorna texto e estatisticas.
    """
    if not text:
        return text, {"dialogue_splits": 0}
    cleaned = text
    total = 0
    for pattern, replacement in DIALOGUE_BREAK_PATTERNS:
        cleaned, count = pattern.subn(replacement, cleaned)
        total += count
    return cleaned, {"dialogue_splits": total}


def strip_stray_triple_quotes(text: str) -> tuple[str, dict]:
    """
    Remove ocorrencias de aspas triplas soltas no final da linha,
    mantendo o restante do conteudo intacto. Retorna texto e estatisticas.
    """
    if not text:
        return text, {"triple_quotes_removed": 0}
    lines: list[str] = []
    total = 0
    for line in text.splitlines(keepends=True):
        newline = ""
        body = line
        if body.endswith("\r\n"):
            newline = "\r\n"
            body = body[:-2]
        elif body.endswith("\n"):
            newline = "\n"
            body = body[:-1]
        elif body.endswith("\r"):
            newline = "\r"
            body = body[:-1]
        body, count = TRIPLE_QUOTES_EOF_RE.subn("", body)
        total += count
        lines.append(body + newline)
    return "".join(lines), {"triple_quotes_removed": total}


def apply_structural_normalizers(text: str) -> tuple[str, dict]:
    """
    Aplica normalizadores estruturais deterministas (dialogo e aspas triplas).
    Retorna texto final e estatisticas simples das correcoes aplicadas.
    """
    cleaned, dialogue_stats = normalize_dialogue_breaks(text)
    cleaned, triple_stats = strip_stray_triple_quotes(cleaned)
    return cleaned, {
        "dialogue_splits": dialogue_stats.get("dialogue_splits", 0),
        "triple_quotes_removed": triple_stats.get("triple_quotes_removed", 0),
    }


def fix_dialogue_artifacts(text: str) -> tuple[str, dict]:
    """
    Corrige artefatos estruturais pós-refine em diálogos.
    Retorna texto corrigido e estatísticas de correções aplicadas.
    """
    stats = {
        "triple_quotes_removed": 0,
        "dialogue_splits": 0,
        "inquote_blank_collapses": 0,
    }

    cleaned, triple_removed = _remove_trailing_triple_quotes(text)
    stats["triple_quotes_removed"] = triple_removed

    cleaned, splits = _split_glued_dialogues(cleaned)
    stats["dialogue_splits"] = splits

    cleaned, collapsed = _collapse_blank_lines_in_quotes(cleaned)
    stats["inquote_blank_collapses"] = collapsed

    return cleaned, stats
