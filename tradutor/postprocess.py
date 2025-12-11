"""
Pós-processamento determinístico para texto refinado em PT-BR.
"""

from __future__ import annotations

import re
from typing import List


def final_pt_postprocess(text: str) -> str:
    """
    Ajustes finais semânticamente neutros:
    - normaliza reticências, travessões e espaços
    - padroniza diálogo com travessão
    - remove marcadores residuais de tradução/refine
    - garante linha vazia entre parágrafos não diálogos
    """
    if not text:
        return text

    cleaned = text
    cleaned = re.sub(r"\.{3,}", "…", cleaned)
    cleaned = cleaned.replace("--", "—")
    cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    cleaned = cleaned.replace(' "', '"').replace(" '", "'")

    # padroniza travessão em diálogos no início da linha
    lines: List[str] = []
    for ln in cleaned.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith("- ") or stripped.startswith("– "):
            ln = ln.replace("- ", "— ", 1) if stripped.startswith("- ") else ln.replace("– ", "— ", 1)
        lines.append(ln)
    cleaned = "\n".join(lines)

    # remove marcadores residuais
    cleaned = re.sub(r"###\s*TEXTO_TRADUZIDO_[A-Z_]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"###\s*TEXTO_REFINADO_[A-Z_]*", "", cleaned, flags=re.IGNORECASE)

    # garante quebra de parágrafo (linha vazia) entre blocos narrativos
    final_lines: List[str] = []
    prev_nonempty = False
    prev_dialog = False
    for ln in cleaned.splitlines():
        stripped = ln.strip()
        if stripped == "":
            final_lines.append("")
            prev_nonempty = False
            prev_dialog = False
            continue
        is_dialog = stripped.startswith("— ")
        if prev_nonempty and not is_dialog and not prev_dialog:
            final_lines.append("")  # insere linha vazia entre parágrafos narrativos consecutivos
        final_lines.append(stripped)
        prev_nonempty = True
        prev_dialog = is_dialog

    result = "\n".join(final_lines).strip()
    return result
