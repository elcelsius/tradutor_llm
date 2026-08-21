from __future__ import annotations

import re
from typing import Dict, List

SECTION_PATTERN = re.compile(
    r"^(?P<title>(?:prologue|epilogue|afterword|chapter\s+\d+(?:(?::|\s*[–—-])\s*[^\n]*)?))\s*$",
    re.IGNORECASE,
)
_SIMPLE_CHAPTER_RE = re.compile(r"^chapter\s+\d+:\s*$", re.IGNORECASE)
_UPPERCASE_WORD_RE = re.compile(r"^[A-Z]{2,}\b")
_SECTION_OPENING_SMALL_CAPS_RE = re.compile(
    r"^(?P<prefix>[A-Z][A-Z'’-]*(?:\s+[A-Z][A-Z'’-]*)*)\s+(?P<rest>[a-zà-ÿ].*)$"
)
_INLINE_POV_SMALL_CAPS_RE = re.compile(
    r"(?P<title>[A-ZÀ-Ý][a-zà-ÿ][a-zà-ÿ'’-]*(?:\s+[A-ZÀ-Ý][a-zà-ÿ][a-zà-ÿ'’-]*){0,4})\s+"
    r"(?P<small_caps>[A-Z][A-Z'’-]*(?:\s+[A-Z][A-Z'’-]*){2,})(?P<rest>.*)$"
)
_SECTION_OPENING_PROSE_WORDS = frozenset(
    {
        "a",
        "am",
        "an",
        "and",
        "are",
        "as",
        "at",
        "began",
        "but",
        "came",
        "can",
        "could",
        "continued",
        "descended",
        "did",
        "do",
        "does",
        "east",
        "felt",
        "finally",
        "for",
        "found",
        "from",
        "had",
        "has",
        "have",
        "he",
        "here",
        "i",
        "in",
        "is",
        "it",
        "knew",
        "looked",
        "mind",
        "must",
        "my",
        "morning",
        "north",
        "next",
        "not",
        "on",
        "or",
        "our",
        "reached",
        "returned",
        "saw",
        "she",
        "should",
        "south",
        "started",
        "stood",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "thought",
        "to",
        "walked",
        "was",
        "we",
        "went",
        "were",
        "west",
        "will",
        "with",
        "without",
        "would",
        "your",
    }
)


def _looks_like_chapter_subtitle(line: str) -> bool:
    """Reconhece um subtítulo curto separado do corpo do capítulo."""
    if not line or line.startswith(("#", '"', "“", "—")):
        return False
    if line.endswith((".", "!", "?")) or line.isupper():
        return False
    words = line.split()
    if not 1 <= len(words) <= 8 or len(line) > 90:
        return False
    return bool(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'’,:;—\- ]+", line)) and line[:1].isupper()


def _chapter_subtitle(body_lines: list[str]) -> tuple[str | None, int]:
    """Retorna o subtítulo e sua linha quando ele precede o corpo do capítulo."""
    idx = 0
    while idx < len(body_lines) and not body_lines[idx].strip():
        idx += 1
    if idx >= len(body_lines):
        return None, idx
    candidate = body_lines[idx].strip()
    if not _looks_like_chapter_subtitle(candidate):
        return None, idx
    next_idx = idx + 1
    while next_idx < len(body_lines) and not body_lines[next_idx].strip():
        next_idx += 1
    if next_idx >= len(body_lines) or SECTION_PATTERN.match(body_lines[next_idx].strip()):
        return None, idx
    return candidate, idx


def _restore_first_person_from_subtitle(
    subtitle: str, body_lines: list[str], subtitle_idx: int
) -> str:
    """Recupera ``I`` deslocado para o fim do subtítulo pela extração do PDF."""
    if not subtitle.endswith(" I"):
        return subtitle
    body_idx = subtitle_idx + 1
    while body_idx < len(body_lines) and not body_lines[body_idx].strip():
        body_idx += 1
    if body_idx >= len(body_lines) or not _UPPERCASE_WORD_RE.match(
        body_lines[body_idx].strip()
    ):
        return subtitle
    body_lines[body_idx] = f"I {body_lines[body_idx].lstrip()}"
    return subtitle[:-2].rstrip()


def _normalize_section_opening_small_caps(body_lines: list[str]) -> None:
    """Converte small caps de prosa na abertura em texto de sentença."""
    for idx, raw_line in enumerate(body_lines):
        line = raw_line.strip()
        if not line:
            continue
        match = _SECTION_OPENING_SMALL_CAPS_RE.match(line)
        if not match:
            return
        words = match.group("prefix").casefold().split()
        if not words or not all(word in _SECTION_OPENING_PROSE_WORDS for word in words):
            return
        normalized_prefix = " ".join(words)
        body_lines[idx] = (
            normalized_prefix[:1].upper()
            + normalized_prefix[1:]
            + " "
            + match.group("rest")
        )
        return


def _split_inline_pov_small_caps(body_lines: list[str]) -> None:
    """Separa POV/timestamp colado à prosa por small caps da extração de PDF."""
    for idx, raw_line in enumerate(body_lines):
        line = raw_line.strip()
        if not line:
            continue
        match = _INLINE_POV_SMALL_CAPS_RE.search(line)
        if not match:
            continue
        words = match.group("small_caps").casefold().split()
        if not words or not all(word in _SECTION_OPENING_PROSE_WORDS for word in words):
            continue
        normalized_opening = " ".join(words)
        normalized_opening = (
            normalized_opening[:1].upper() + normalized_opening[1:]
        )
        prefix = line[: match.start()].rstrip()
        heading_block = (
            f"## {match.group('title')}\n\n{normalized_opening}{match.group('rest')}"
        )
        body_lines[idx] = f"{prefix}\n\n{heading_block}" if prefix else heading_block


def _is_toc_stub_body(body: str) -> bool:
    """Processamento interno auxiliar."""
    stripped = body.strip()
    if not stripped:
        return True
    if len(stripped) <= 10 and re.fullmatch(r"[\d\s.]+", stripped):
        return True
    if len(stripped.split()) <= 2 and not re.search(r"[A-Za-zÀ-ÿ]", stripped):
        return True
    return False


def split_into_sections(text: str) -> List[Dict]:
    """
    Divide texto bruto em seções por marcadores de capítulo.
    Retorna lista de dicts: {"title": str, "body": str, "start_idx": int, "end_idx": int}
    Se nenhum marcador for encontrado, retorna uma única seção "Full Text".
    """
    lines = text.splitlines()
    matches = []
    for idx, ln in enumerate(lines):
        if SECTION_PATTERN.match(ln.strip()):
            matches.append((idx, ln.strip()))

    if not matches:
        return [
            {
                "title": "Full Text",
                "body": text.strip(),
                "start_idx": 0,
                "end_idx": len(text),
            }
        ]

    sections: List[Dict] = []
    first_start = matches[0][0]
    if first_start > 0:
        pre_body = "\n".join(lines[:first_start]).strip()
        if pre_body:
            pre_end_idx = sum(len(line) + 1 for line in lines[:first_start])
            sections.append(
                {
                    "title": "Full Text",
                    "body": pre_body,
                    "start_idx": 0,
                    "end_idx": pre_end_idx,
                }
            )
    for i, (start_line, title) in enumerate(matches):
        end_line = matches[i + 1][0] if i + 1 < len(matches) else len(lines)
        body_lines = lines[start_line + 1 : end_line]
        header = f"# {title}"
        if _SIMPLE_CHAPTER_RE.fullmatch(title):
            subtitle, subtitle_idx = _chapter_subtitle(body_lines)
            if subtitle:
                subtitle = _restore_first_person_from_subtitle(
                    subtitle, body_lines, subtitle_idx
                )
                # A marcação separa título e narrativa no prompt. Sem ela, uma
                # LLM pode colar o subtítulo à primeira frase do capítulo.
                header = f"{header}\n\n## {subtitle}"
                body_lines = body_lines[subtitle_idx + 1 :]
        _split_inline_pov_small_caps(body_lines)
        _normalize_section_opening_small_caps(body_lines)
        body = "\n".join(body_lines).strip()
        if _is_toc_stub_body(body):
            # Possível entrada de sumário; ignora se o corpo é vazio/curto ou numérico.
            continue
        full_body = f"{header}\n\n{body}".strip()
        start_idx = sum(len(line) + 1 for line in lines[:start_line])  # approx byte offset
        end_idx = sum(len(line) + 1 for line in lines[:end_line])
        sections.append(
            {
                "title": title,
                "body": full_body,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )
    return sections
