"""
Normalização leve da estrutura (títulos/capítulos) antes do PDF.
"""

from __future__ import annotations

import re
from typing import List


def normalize_structure(text: str) -> str:
    lines = text.splitlines()
    normalized: List[str] = []
    heading_re = re.compile(
        r"^(?P<head>(?:pr[oó]logo|cap[ií]tulo\s+[^\s].*?|ep[ií]logo|interl[úu]dio))(?P<rest>.*)$",
        re.IGNORECASE,
    )

    def add_blank() -> None:
        if normalized and normalized[-1] == "":
            return
        normalized.append("")

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            add_blank()
            continue

        m = heading_re.match(stripped.rstrip(":"))
        if m:
            head = m.group("head").strip()
            rest = m.group("rest").strip()
            normalized.append(head)
            add_blank()
            if rest:
                normalized.append(rest)
                add_blank()
            continue

        normalized.append(stripped)

    # remove blanks duplicados no final/início
    cleaned: List[str] = []
    last_blank = False
    for ln in normalized:
        if ln == "":
            if last_blank:
                continue
            last_blank = True
        else:
            last_blank = False
        cleaned.append(ln)

    return "\n".join(cleaned).strip()
