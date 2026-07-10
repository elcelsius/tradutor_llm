from __future__ import annotations

import re

ENGLISH_LEAK_WORDS = frozenset(
    """
    about after again against all also although always among another around asked
    because before behind believe between brought called calmly came choose choosing
    could desire directly does doesn't doing done enough every everyone everything
    felt from gave gives going gone have haven't having however into itself knew
    know knowing made make makes might must neither never nothing once only other
    perhaps replied said says should something still strongly such than that their
    theirs them themselves then there these they they're this those though through
    told toward under until upon wanted wasn't were weren't what whatever when
    where whether which while who whom whose with within without would wouldn't
    your you're yours
    """.split()
)
PORTUGUESE_ANCHOR_WORDS = frozenset(
    """
    agora ainda alguém algum alguma alguns algumas antes aqui assim até caso com
    como da das de dela dele deles depois do dos ela elas ele eles em então era
    essa esse isso mais mas mesmo minha muito na não nas nem no nos para pela pelo
    por porque quando que quem sem seu sua talvez também tinha uma você vocês
    """.split()
)
ENGLISH_CONTRACTION_RE = re.compile(
    r"\b(?:i['’]m|i['’]ll|i['’]ve|you['’]re|you['’]ll|they['’]re|we['’]re|"
    r"it['’]s|that['’]s|can['’]t|won['’]t|don['’]t|doesn['’]t|didn['’]t|"
    r"wouldn['’]t|shouldn['’]t|couldn['’]t|haven['’]t|weren['’]t|wasn['’]t)\b",
    re.IGNORECASE,
)
ENGLISH_POSSESSIVE_RE = re.compile(r"\b[A-Z][A-Za-z]+['’]s\b")


def english_leak_segments(text: str) -> list[str]:
    """Detecta segmentos longos que ainda parecem estar em ingles."""
    if not text:
        return []
    flagged: list[str] = []
    segments = re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
    for raw_segment in segments:
        segment = raw_segment.strip()
        if len(segment) < 35:
            continue
        words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", segment)
        if len(words) < 6:
            continue
        normalized = [word.lower().replace("’", "'") for word in words]
        english_hits = sum(1 for word in normalized if word in ENGLISH_LEAK_WORDS)
        english_hits += len(ENGLISH_CONTRACTION_RE.findall(segment)) * 2
        english_hits += len(ENGLISH_POSSESSIVE_RE.findall(segment))
        portuguese_hits = sum(1 for word in normalized if word in PORTUGUESE_ANCHOR_WORDS)
        dense_english = len(words) >= 10 and english_hits >= 5 and english_hits / max(len(words), 1) >= 0.25
        dominant_english = english_hits >= 4 and english_hits >= portuguese_hits + 2
        if dense_english or dominant_english:
            flagged.append(segment)
    return flagged


def detect_residual_english(text: str) -> tuple[bool, str]:
    segments = english_leak_segments(text)
    if not segments:
        return False, ""
    preview = re.sub(r"\s+", " ", segments[0]).strip()
    if len(preview) > 80:
        preview = preview[:77].rstrip() + "..."
    return True, f"residual_english:{preview}"
