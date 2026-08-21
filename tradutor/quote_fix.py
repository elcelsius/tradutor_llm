from __future__ import annotations

import logging
import re
from typing import Tuple

NARRATION_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?:Ele|Ela|Eles|Elas|Ayaka|Banewolf|Agit|Abis|Kirihara|Oyamada)\b"
)
_REPORTING_VERB_PATTERN = re.compile(
    r"\b(?:afirmou|comentou|disse|exclamou|falou|gritou|indagou|murmurou|"
    r"observou|perguntou|respondeu|retrucou)\b",
    flags=re.IGNORECASE,
)


def count_curly_quotes(text: str) -> Tuple[int, int]:
    """Conta aspas curvas de abertura/fechamento."""
    return text.count("“"), text.count("”")


def collapse_repeated_curly_quotes(text: str) -> Tuple[str, int]:
    """Colapsa sequencias espurias como ``””`` ou ``““`` em uma aspa.

    A regra nao toca em ``”“``, que representa uma fronteira entre falas e e
    validada separadamente pelos guardrails de dialogo.
    """
    return re.subn(r"([“”])\1+", r"\1", text)


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


def _first_unmatched_close(text: str) -> int | None:
    """Retorna a primeira aspa de fechamento sem abertura correspondente."""
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "“":
            depth += 1
        elif ch == "”":
            if depth:
                depth -= 1
            else:
                return idx
    return None


def _safe_missing_open_insert_position(text: str, unmatched_close: int) -> int | None:
    """Encontra o inicio da fala quando um fechamento isolado e inequivoco aparece.

    So corrigimos uma linha que ja possui texto suficiente antes do fechamento e
    nao contem uma abertura curva. Isso cobre o caso comum em que uma fala foi
    dividida pelo PDF/LLM e perdeu apenas a aspa de abertura, sem tentar adivinhar
    a estrutura de uma linha composta apenas por narracao.
    """
    line_start = text.rfind("\n", 0, unmatched_close) + 1
    prefix = text[line_start:unmatched_close]
    stripped_prefix = prefix.strip()
    if "“" in prefix:
        return None
    if len(stripped_prefix) < 8:
        # Uma nota musical isolada seguida de fechamento (``♪”``) é uma fala
        # curta que perdeu a abertura na extração. Não abrimos aspas para
        # outros fragmentos curtos, pois seriam ambíguos.
        if stripped_prefix != "♪":
            return None
    leading = len(prefix) - len(prefix.lstrip())
    return line_start + leading


def _safe_stray_close_remove_position(text: str, unmatched_close: int) -> int | None:
    """Identifica um fechamento extra apos uma fala ja encerrada e narracao.

    Um caso recorrente da LLM e fechar novamente, no fim do paragrafo, uma fala
    que ja foi fechada antes de uma frase de narracao. Nao removemos uma aspa
    apenas pela contagem global: exigimos uma fala local balanceada, um verbo de
    elocucao e pelo menos uma segunda frase de narracao antes do fechamento.
    """
    line_start = text.rfind("\n", 0, unmatched_close) + 1
    prefix = text[line_start:unmatched_close]
    if not prefix or prefix.count("“") != prefix.count("”") or "“" not in prefix:
        return None

    last_close = prefix.rfind("”")
    narration_tail = prefix[last_close + 1 :].strip()
    if len(narration_tail) < 24:
        return None
    if not _REPORTING_VERB_PATTERN.search(narration_tail):
        return None
    if not re.search(r"[.!?…]\s+\S", narration_tail):
        return None
    return unmatched_close


def repair_missing_open_quotes_per_paragraph(
    text: str,
    logger: logging.Logger | None = None,
    label: str | None = None,
) -> Tuple[str, int]:
    """Restaura aberturas perdidas mesmo quando o total global parece balanceado.

    A extração ou a desquebra pode perder a abertura de uma fala em um
    parágrafo e preservar outras aspas em posições diferentes. Nesse caso as
    contagens globais podem se compensar, mas a fala continua inválida. Só
    alteramos parágrafos com uma única aspa de fechamento sem abertura local e
    com uma posição de inserção inequivoca no início da linha.
    """
    if not text or "”" not in text:
        return text, 0

    parts = re.split(r"(\n\s*\n)", text)
    fixes = 0
    for index in range(0, len(parts), 2):
        paragraph = parts[index]
        if not paragraph or "”" not in paragraph:
            continue

        depth = 0
        unmatched_closes: list[int] = []
        for position, char in enumerate(paragraph):
            if char == "“":
                depth += 1
            elif char == "”":
                if depth:
                    depth -= 1
                else:
                    unmatched_closes.append(position)

        if depth or len(unmatched_closes) != 1:
            continue
        insert_pos = _safe_missing_open_insert_position(paragraph, unmatched_closes[0])
        if insert_pos is None:
            continue
        parts[index] = paragraph[:insert_pos] + "“" + paragraph[insert_pos:]
        fixes += 1

    if fixes and logger:
        logger.info(
            "Aberturas de diálogo restauradas%s: %d",
            f" ({label})" if label else "",
            fixes,
        )
    return "".join(parts), fixes


def repair_missing_closing_curly_quotes_per_paragraph(
    text: str,
    logger: logging.Logger | None = None,
    label: str | None = None,
) -> Tuple[str, int]:
    """Converte um fechamento reto terminal em ``”`` quando ele é inequívoco.

    Alguns PDFs preservam a abertura curva de uma fala, mas extraem o
    fechamento final como ``\"``. A conversão só ocorre quando o parágrafo tem
    exatamente uma abertura curva pendente, nenhum fechamento órfão e termina
    na aspa reta. Assim, medidas como ``6\"`` e aspas inline normais não são
    alteradas.
    """
    if not text or "“" not in text or '"' not in text:
        return text, 0

    parts = re.split(r"(\n\s*\n)", text)
    fixes = 0
    for index in range(0, len(parts), 2):
        paragraph = parts[index]
        stripped_end = len(paragraph.rstrip())
        if not paragraph or not stripped_end or paragraph[stripped_end - 1] != '"':
            continue

        depth = 0
        has_unmatched_close = False
        for char in paragraph:
            if char == "“":
                depth += 1
            elif char == "”":
                if depth:
                    depth -= 1
                else:
                    has_unmatched_close = True
                    break

        if depth != 1 or has_unmatched_close:
            continue
        parts[index] = (
            paragraph[: stripped_end - 1] + "”" + paragraph[stripped_end:]
        )
        fixes += 1

    if fixes and logger:
        logger.info(
            "Fechamentos de diálogo normalizados%s: %d",
            f" ({label})" if label else "",
            fixes,
        )
    return "".join(parts), fixes


def fix_unbalanced_quotes(
    text: str, logger: logging.Logger | None = None, label: str | None = None
) -> Tuple[str, bool]:
    """
    Se houver exatamente uma aspa curva faltando, tenta restaurar a contraparte.
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

    if opens - closes == 1:
        unmatched = _first_unmatched_open(text)
        if unmatched is None:
            return text, False

        next_open = text.find("“", unmatched + 1)
        search_end = next_open if next_open != -1 else len(text)
        segment = text[unmatched:search_end]
        match = NARRATION_PATTERN.search(segment)

        if match:
            insert_pos = unmatched + match.start()
        elif next_open != -1:
            insert_pos = next_open
        else:
            insert_pos = len(text)

        fixed = text[:insert_pos] + "”" + text[insert_pos:]
        return fixed, True

    if closes - opens == 1:
        unmatched = _first_unmatched_close(text)
        if unmatched is None:
            return text, False
        remove_pos = _safe_stray_close_remove_position(text, unmatched)
        if remove_pos is not None:
            fixed = text[:remove_pos] + text[remove_pos + 1 :]
            return fixed, True
        insert_pos = _safe_missing_open_insert_position(text, unmatched)
        if insert_pos is None:
            return text, False
        fixed = text[:insert_pos] + "“" + text[insert_pos:]
        return fixed, True

    return text, False


def fix_blank_lines_inside_quotes(
    text: str, logger: logging.Logger | None = None, label: str | None = None
) -> Tuple[str, int]:
    """
    Remove parágrafos em branco dentro de blocos entre “ e ”.
    Converte \\n\\s*\\n para um único \\n quando in_quote.
    """
    cleaned, fixes = _collapse_blank_lines_in_quotes(text)
    if fixes and logger:
        logger.debug(
            "Correção de linhas em branco dentro de aspas%s: %d",
            f" ({label})" if label else "",
            fixes,
        )
    return cleaned, fixes


def _collapse_blank_lines_in_quotes(text: str) -> Tuple[str, int]:
    """Colapsa parágrafos artificiais apenas quando dentro de aspas curvas."""
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
                # Uma mesma fala não deve conter um parágrafo vazio. Se a
                # próxima coisa for o fechamento, una-o à frase anterior;
                # caso contrário, transforme a quebra artificial em espaço.
                if j < length and text[j] != "”":
                    if cleaned and not cleaned[-1].isspace():
                        cleaned.append(" ")
                fixes += blank_lines
                i = j
                continue
        cleaned.append(ch)
        i += 1
    return "".join(cleaned), fixes
