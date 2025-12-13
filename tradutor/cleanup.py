"""
Rotinas determinísticas de limpeza estrutural antes do refine.
"""

from __future__ import annotations

import re


def _normalize_spaces(line: str) -> str:
    return " ".join(line.split())


def dedupe_adjacent_lines(text: str) -> tuple[str, dict]:
    """
    Remove linhas/parágrafos consecutivos idênticos (ignorando múltiplos espaços).
    Retorna texto limpo e stats {"lines_removed": int, "blocks_removed": int}.
    """
    lines = text.splitlines()
    deduped: list[str] = []
    prev_norm: str | None = None
    lines_removed = 0
    for ln in lines:
        norm = _normalize_spaces(ln)
        if norm and prev_norm == norm:
            lines_removed += 1
            continue
        deduped.append(ln)
        prev_norm = norm if norm else None

    paragraphs = "\n".join(deduped).split("\n\n")
    final_paragraphs: list[str] = []
    prev_para_norm: str | None = None
    blocks_removed = 0
    for para in paragraphs:
        norm_para = _normalize_spaces(para.strip())
        if norm_para and prev_para_norm == norm_para:
            blocks_removed += 1
            continue
        final_paragraphs.append(para)
        prev_para_norm = norm_para if norm_para else None

    return "\n\n".join(final_paragraphs), {"lines_removed": lines_removed, "blocks_removed": blocks_removed}


_GLUED_PATTERN = re.compile(r'([.!?]["”\']?)\s+("?[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ])')


def fix_glued_dialogues(text: str) -> tuple[str, dict]:
    """
    Insere quebras conservadoras quando duas falas/sentenças estão coladas
    no mesmo parágrafo sem separação evidente.
    """
    fixed_lines: list[str] = []
    breaks_inserted = 0
    for ln in text.splitlines():
        if ln.lstrip().startswith("#"):
            fixed_lines.append(ln)
            continue
        new_line, count = _GLUED_PATTERN.subn(r"\1\n\2", ln)
        breaks_inserted += count
        fixed_lines.append(new_line)
    return "\n".join(fixed_lines), {"breaks_inserted": breaks_inserted}


def dedupe_prefix_lines(text: str) -> tuple[str, dict]:
    """
    Remove linha truncada quando a próxima começa com ela e termina aberta.

    Considera pontuação forte: . ! ? … : ;
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    removed = 0

    def _ends_open(ln: str) -> bool:
        return not re.search(r"[.!?…:;]['\")\]]?\s*$", ln)

    idx = 0
    total = len(lines)
    while idx < total:
        current = lines[idx]
        nxt = lines[idx + 1] if idx + 1 < total else None
        if nxt is not None:
            cur_strip = current.strip()
            nxt_strip = nxt.strip()
            cur_norm = _normalize_spaces(cur_strip)
            nxt_norm = _normalize_spaces(nxt_strip)
            if (
                cur_norm
                and nxt_norm
                and len(nxt_norm) > len(cur_norm)
                and nxt_norm.startswith(cur_norm)
                and _ends_open(cur_norm)
            ):
                removed += 1
                idx += 1
                continue
        cleaned.append(current)
        idx += 1

    return "\n".join(cleaned), {"prefix_lines_removed": removed}


def cleanup_before_refine(md: str) -> tuple[str, dict]:
    """
    Executa passos determinísticos antes do refine.
    Retorna (md_limpo, stats combinados).
    """
    deduped, stats_dedupe = dedupe_adjacent_lines(md)
    prefix_cleaned, stats_prefix = dedupe_prefix_lines(deduped)
    fixed, stats_glued = fix_glued_dialogues(prefix_cleaned)
    combined = {
        "lines_removed": stats_dedupe.get("lines_removed", 0),
        "blocks_removed": stats_dedupe.get("blocks_removed", 0),
        "breaks_inserted": stats_glued.get("breaks_inserted", 0),
        "prefix_lines_removed": stats_prefix.get("prefix_lines_removed", 0),
    }
    return fixed, combined


def detect_obvious_dupes(md: str) -> bool:
    """
    Heurística leve para detectar duplicações adjacentes.
    Dispara se encontrar pelo menos duas repetições consecutivas.
    """
    lines = [ln for ln in md.splitlines() if _normalize_spaces(ln)]
    consecutive = 0
    prev = None
    for ln in lines:
        norm = _normalize_spaces(ln)
        if prev is not None and norm == prev:
            consecutive += 1
            if consecutive >= 2:
                return True
        else:
            consecutive = 0
        prev = norm
    return False


def detect_glued_dialogues(md: str) -> bool:
    """
    Heurística leve para detectar falas coladas.
    """
    return bool(_GLUED_PATTERN.search(md))
