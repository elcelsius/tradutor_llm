from __future__ import annotations

import re
from typing import Tuple

from .quote_fix import _collapse_blank_lines_in_quotes

TRIPLE_QUOTES_EOF_RE = re.compile(r'"""+\s*$')
DIALOGUE_GLUE_MULTI_RE = re.compile(r"”{2,}[ \t]*“")
DIALOGUE_GLUE_RE = re.compile(r"”[ \t]*“")


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
    cleaned, count_multi = DIALOGUE_GLUE_MULTI_RE.subn("”\n\n“", text)
    cleaned, count_single = DIALOGUE_GLUE_RE.subn("”\n\n“", cleaned)
    return cleaned, count_multi + count_single


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
