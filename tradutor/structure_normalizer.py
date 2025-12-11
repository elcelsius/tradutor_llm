"""
Normalização leve da estrutura (títulos/capítulos) antes do PDF.
"""

from __future__ import annotations

import re
from typing import List


def normalize_structure(text: str) -> str:
    lines = text.splitlines()
    normalized: List[str] = []
    chapter_re = re.compile(r"^(?:chapter|cap[ií]tulo)\s+(\d+)", re.IGNORECASE)
    last_blank = False

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            if not last_blank:
                normalized.append("")
            last_blank = True
            continue
        last_blank = False
        m = chapter_re.match(stripped.rstrip(":"))
        if m:
            num = m.group(1)
            normalized.append(f"CAPÍTULO {num}")
            normalized.append("")  # separa do parágrafo seguinte
            continue
        normalized.append(stripped)

    # remove linhas duplicadas de título consecutivo
    deduped: List[str] = []
    prev = None
    for ln in normalized:
        if ln and ln == prev and ln.startswith("CAPÍTULO"):
            continue
        deduped.append(ln)
        prev = ln
    return "\n".join(deduped).strip()
