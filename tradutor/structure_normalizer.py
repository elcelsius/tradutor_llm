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

        split_chapter = _split_deathmatch_chapter_title(stripped)
        if split_chapter:
            title, body = split_chapter
            previous_heading_idx = _last_nonblank_index(normalized)
            if previous_heading_idx is not None and re.match(
                r"^#?\s*cap[ií]tulo\s+1:\s*$",
                normalized[previous_heading_idx].strip(),
                flags=re.IGNORECASE,
            ):
                prefix = "# " if normalized[previous_heading_idx].lstrip().startswith("#") else ""
                normalized[previous_heading_idx] = f"{prefix}Capítulo 1: {title}"
                add_blank()
                normalized.append(body)
                add_blank()
                continue
            if previous_heading_idx is None:
                normalized.append(f"# Capítulo 1: {title}")
                add_blank()
                normalized.append(body)
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


def _last_nonblank_index(lines: list[str]) -> int | None:
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            return idx
    return None


def _split_deathmatch_chapter_title(line: str) -> tuple[str, str] | None:
    patterns = [
        r"^(?:Depois do Deathmatch|Depois do Combate Mortal)\s+(?P<body>(?:DEPOIS QUE|Depois que)\b.*)$",
        r"^Após o combate mortal,?\s+(?P<body>(?:após|Após)\b.*)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, line, flags=re.IGNORECASE)
        if not match:
            continue
        body = match.group("body").strip()
        body = re.sub(r"^DEPOIS QUE\s+", "Depois que ", body)
        body = re.sub(r"^após\s+", "Após ", body, flags=re.IGNORECASE)
        body = re.sub(r"\bSOGOU\b", "Sogou", body)
        return "Após o Combate Mortal", body
    return None
