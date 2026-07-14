"""
Normalização leve da estrutura (títulos, capítulos e marcadores de cena).
"""

from __future__ import annotations

import re
from typing import List


_SIMPLE_CHAPTER_RE = re.compile(
    r"^(?:#\s*)?cap[ií]tulo\s+(?P<number>\d+):?\s*$",
    re.IGNORECASE,
)
_MARKDOWN_CHAPTER_WITH_SUBTITLE_RE = re.compile(
    r"^#\s*cap[ií]tulo\s+(?P<number>\d+):\s+(?P<subtitle>\S.+?)\s*$",
    re.IGNORECASE,
)
_MARKDOWN_SIMPLE_CHAPTER_RE = re.compile(
    r"^#\s*cap[ií]tulo\s+(?P<number>\d+):?\s*$",
    re.IGNORECASE,
)
_TIME_LABEL_RE = re.compile(
    r"^(?P<name>[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+){1,3})\s+"
    r"(?P<label>ALGUM TEMPO ANTES(?:,\s*há um tempo)?…?)$"
)


def normalize_structure(text: str) -> str:
    """Recupera estrutura que pode ter sido colada pela extração ou por uma LLM.

    A função só atua em padrões estruturais inequívocos. Ela não reorganiza
    parágrafos narrativos nem tenta reescrever conteúdo literário.
    """
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

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            add_blank()
            i += 1
            continue

        # O revisor de headings pode inserir ``# Capitulo N:`` mesmo quando a
        # LLM ja gerou ``# Capitulo N: subtitulo``. Mantemos o mais informativo.
        titled_chapter = _MARKDOWN_CHAPTER_WITH_SUBTITLE_RE.match(stripped)
        if titled_chapter:
            next_idx = i + 1
            while next_idx < len(lines) and not lines[next_idx].strip():
                next_idx += 1
            duplicate = (
                _MARKDOWN_SIMPLE_CHAPTER_RE.match(lines[next_idx].strip())
                if next_idx < len(lines)
                else None
            )
            if duplicate and duplicate.group("number") == titled_chapter.group("number"):
                normalized.append(
                    f"# Capítulo {titled_chapter.group('number')}: {titled_chapter.group('subtitle')}"
                )
                add_blank()
                i = next_idx + 1
                continue

        if re.match(r"^DEPOIS QUE\b", stripped):
            stripped = re.sub(r"^DEPOIS QUE\s+", "Depois que ", stripped)
            stripped = re.sub(r"\bSOGOU\b", "Sogou", stripped)

        if stripped.startswith("***") and stripped != "***":
            normalized.append("***")
            add_blank()
            normalized.append(stripped[3:].strip())
            add_blank()
            i += 1
            continue

        split_chapter = _split_deathmatch_chapter_title(stripped)
        if split_chapter:
            title, body = split_chapter
            previous_heading_idx = _last_nonblank_index(normalized)
            if previous_heading_idx is not None and _SIMPLE_CHAPTER_RE.match(normalized[previous_heading_idx].strip()):
                normalized[previous_heading_idx] = f"# Capítulo 1: {title}"
                add_blank()
                normalized.append(body)
                add_blank()
                i += 1
                continue
            if previous_heading_idx is None:
                normalized.append(f"# Capítulo 1: {title}")
                add_blank()
                normalized.append(body)
                add_blank()
                i += 1
                continue

        chapter_match = _SIMPLE_CHAPTER_RE.match(stripped)
        if chapter_match:
            subtitle, subtitle_idx = _next_chapter_subtitle(lines, i + 1)
            if subtitle:
                normalized.append(f"# Capítulo {chapter_match.group('number')}: {subtitle}")
                add_blank()
                i = subtitle_idx + 1
                continue
            normalized.append(f"# Capítulo {chapter_match.group('number')}:")
            add_blank()
            i += 1
            continue

        time_label = _split_character_time_label(stripped)
        if time_label:
            name, label = time_label
            normalized.append(f"## {name}")
            add_blank()
            normalized.append(label)
            add_blank()
            i += 1
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
            i += 1
            continue

        normalized.append(stripped)
        i += 1

    # Remove blanks duplicados no final/início.
    cleaned: List[str] = []
    last_blank = False
    for line in normalized:
        if line == "":
            if last_blank:
                continue
            last_blank = True
        else:
            last_blank = False
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _last_nonblank_index(lines: list[str]) -> int | None:
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            return idx
    return None


def _next_chapter_subtitle(lines: list[str], start: int) -> tuple[str | None, int]:
    """Retorna um subtítulo curto entre heading de capítulo e corpo narrativo."""
    idx = start
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return None, idx
    candidate = lines[idx].strip()
    if not _looks_like_chapter_subtitle(candidate):
        return None, idx

    next_idx = idx + 1
    while next_idx < len(lines) and not lines[next_idx].strip():
        next_idx += 1
    if next_idx >= len(lines):
        return None, idx
    return candidate, idx


def _looks_like_chapter_subtitle(line: str) -> bool:
    if not line or line.startswith(("#", "\"", "“", "—")):
        return False
    if line.endswith((".", "!", "?")) or line.isupper():
        return False
    words = line.split()
    if not 2 <= len(words) <= 8 or len(line) > 90:
        return False
    return bool(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'’,:;—\- ]+", line)) and line[:1].isupper()


def _split_character_time_label(line: str) -> tuple[str, str] | None:
    match = _TIME_LABEL_RE.match(line)
    if not match:
        return None
    return match.group("name"), "Algum tempo antes…"


def _split_deathmatch_chapter_title(line: str) -> tuple[str, str] | None:
    patterns = [
        r"^(?:Depois do Deathmatch|Depois do Combate Mortal)\s+(?P<body>(?:DEPOIS QUE|Depois que)\b.*)$",
        r"^Após o (?:Deathmatch|combate mortal|Combate Mortal),?\s+(?P<body>(?:depois que|Depois que|após|Após)\b.*)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, line, flags=re.IGNORECASE)
        if not match:
            continue
        body = match.group("body").strip()
        body = re.sub(r"^DEPOIS QUE\s+", "Depois que ", body)
        body = re.sub(r"^depois que\s+", "Depois que ", body, flags=re.IGNORECASE)
        body = re.sub(r"^após\s+", "Após ", body, flags=re.IGNORECASE)
        body = re.sub(r"\bSOGOU\b", "Sogou", body)
        return "Após o Combate Mortal", body
    return None
