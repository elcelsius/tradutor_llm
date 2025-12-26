from __future__ import annotations

import logging
import re
from typing import Tuple

NARRATION_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?:Ele|Ela|Eles|Elas|Ayaka|Banewolf|Agit|Abis|Kirihara|Oyamada)\b"
)


def count_curly_quotes(text: str) -> Tuple[int, int]:
    """Conta aspas curvas de abertura/fechamento."""
    return text.count("“"), text.count("”")


def _first_unmatched_open(text: str) -> int | None:
    """Retorna índice da primeira aspa de abertura não fechada (ou None)."""
    stack: list[int] = []
    for idx, ch in enumerate(text):
        if ch == "“":
            stack.append(idx)
        elif ch == "”":
            if stack:
                stack.pop()
    return stack[0] if stack else None


def fix_unbalanced_quotes(text: str, logger: logging.Logger | None = None, label: str | None = None) -> Tuple[str, bool]:
    """
    Se houver exatamente uma aspa de abertura a mais, tenta inserir a aspa de fechamento.
    Retorna (texto_corrigido, alterado).
    """
    opens, closes = count_curly_quotes(text)
    if opens == closes:
        return text, False

    if logger:
        logger.warning(
            "Aspas curvas desbalanceadas%s: %d “ vs %d ”",
            f" ({label})" if label else "",
            opens,
            closes,
        )

    if opens - closes != 1:
        return text, False

    unmatched = _first_unmatched_open(text)
    if unmatched is None:
        return text, False

    next_open = text.find("“", unmatched + 1)
    search_end = next_open if next_open != -1 else len(text)
    segment = text[unmatched:search_end]
    match = NARRATION_PATTERN.search(segment)

    insert_pos = None
    if match:
        insert_pos = unmatched + match.start()
    elif next_open != -1:
        insert_pos = next_open
    else:
        insert_pos = len(text)

    fixed = text[:insert_pos] + "”" + text[insert_pos:]
    return fixed, True


def fix_blank_lines_inside_quotes(text: str, logger: logging.Logger | None = None, label: str | None = None) -> Tuple[str, int]:
    """
    Remove parágrafos em branco dentro de blocos entre “ e ”.
    Converte \\n\\s*\\n para um único \\n quando in_quote.
    """
    cleaned, fixes = _collapse_blank_lines_in_quotes(text)
    if fixes and logger:
        logger.debug("Correção de linhas em branco dentro de aspas%s: %d", f" ({label})" if label else "", fixes)
    return cleaned, fixes


def _collapse_blank_lines_in_quotes(text: str) -> Tuple[str, int]:
    """Colapsa linhas em branco apenas quando dentro de aspas curvas."""
    in_quote = False
    i = 0
    cleaned: list[str] = []
    fixes = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == "“":
            in_quote = True
            cleaned.append(ch)
            i += 1
            continue
        if ch == "”":
            in_quote = False
            cleaned.append(ch)
            i += 1
            continue
        if in_quote and ch == "\n":
            whitespace_start = i + 1
            j = whitespace_start
            blank_lines = 0
            while True:
                while j < length and text[j] in " \t":
                    j += 1
                if j < length and text[j] == "\n":
                    blank_lines += 1
                    j += 1
                    whitespace_start = j
                    continue
                break
            if blank_lines:
                cleaned.append("\n")
                indent = text[whitespace_start:j]
                if indent:
                    cleaned.append(indent)
                fixes += blank_lines
                i = j
                continue
        cleaned.append(ch)
        i += 1
    return "".join(cleaned), fixes


def fix_dialogue_artifacts(text: str, logger: logging.Logger | None = None, label: str | None = None) -> str:
    """
    Corrige artefatos estruturais de diálogo pós-refine de forma determinística.
    - Separa falas coladas (“ ” ou ”“) em parágrafos distintos.
    - Colapsa parágrafos em branco dentro de uma mesma fala.
    """
    cleaned, dialogue_splits = re.subn(r"”\s*“", "”\n\n“", text)
    cleaned, collapsed_blanks = _collapse_blank_lines_in_quotes(cleaned)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "fix_dialogue_artifacts%s: splits=%d blank_lines_collapsed=%d",
            f" ({label})" if label else "",
            dialogue_splits,
            collapsed_blanks,
        )
    return cleaned
