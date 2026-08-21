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
# Formas plurais inglesas que não têm uso natural em PT-BR. Mantemos a lista
# curta para não sinalizar empréstimos técnicos legítimos ou nomes próprios.
SINGLE_TOKEN_ENGLISH_LEAKS = frozenset(
    {
        "af",
        "and",
        "arright",
        "boost",
        "but",
        "buff",
        "buffs",
        "kys",
        "selves",
        "they",
        "though",
        "uh",
        "uhh",
        "or",
    }
)
# Inclui a forma hibrida que alguns modelos produzem ao preservar "I see"
# e traduzir apenas o pronome.
SHORT_ENGLISH_LEAKS = frozenset({"i see", "eu see"})
EMBEDDED_ENGLISH_TRIGGER_WORDS = frozenset(
    {
        "all",
        "and",
        "are",
        "but",
        "for",
        "from",
        "has",
        "have",
        "is",
        "it",
        "same",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
EMBEDDED_ENGLISH_WORDS = (
    ENGLISH_LEAK_WORDS | EMBEDDED_ENGLISH_TRIGGER_WORDS | frozenset(
    {
        "be",
        "been",
        "command",
        "did",
        "does",
        "goddess",
        "has",
        "he",
        "i",
        "is",
        "same",
        "say",
        "said",
        "says",
        "she",
        "the",
        "trust",
        "trusts",
        "was",
        "were",
        "will",
    }
    )
)
# ``I-impossível`` é uma gagueira válida em português, enquanto
# ``I-impossible`` ainda é um resíduo em inglês. Portanto, só sinalizamos a
# forma hifenizada quando a palavra seguinte é inequivocamente inglesa.
_STUTTERED_ENGLISH_I_WORD_RE = (
    r"(?:i(?:['’](?:m|ll|ve|d|t|s))?|am|are|is|was|were|will|would|"
    r"can(?:not)?|could|do(?:n't)?|does(?:n't)?|did(?:n't)?|"
    r"have(?:n't)?|has|had|know|knew|think|thought|want|need|mean|"
    r"guess|hope|feel|see|saw|tell|told|try|tried|go|went|come|came|"
    r"should|must|might|may|impossible)\b"
)
MIXED_ENGLISH_ARTIFACT_RE = re.compile(
    rf"(?<![A-Za-zÀ-ÿ])(?:I\s+(?=[a-zà-ÿ])|"
    rf"Eu\s+{_STUTTERED_ENGLISH_I_WORD_RE}|"
    rf"I-(?={_STUTTERED_ENGLISH_I_WORD_RE})|Y-you\b)",
    re.IGNORECASE,
)


def _embedded_english_window(text: str) -> str:
    """Encontra uma curta sequencia inglesa inserida em frase em portugues."""
    tokens = list(re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text))
    normalized = [match.group(0).lower().replace("’", "'") for match in tokens]
    min_window = 5
    max_window = 8
    for start in range(len(tokens)):
        if normalized[start] not in EMBEDDED_ENGLISH_WORDS:
            continue
        upper_bound = min(len(tokens), start + max_window)
        for end in range(upper_bound, start + min_window - 1, -1):
            window = normalized[start:end]
            english_hits = sum(word in EMBEDDED_ENGLISH_WORDS for word in window)
            if (
                english_hits >= 4
                and english_hits / len(window) >= 0.5
                and any(word in EMBEDDED_ENGLISH_TRIGGER_WORDS for word in window)
            ):
                # A janela pode conter palavras PT-BR logo depois do vazamento.
                # Retire-as para dar ao reparo o trecho inglês mais preciso.
                trimmed_end = end
                while (
                    trimmed_end > start
                    and normalized[trimmed_end - 1] not in EMBEDDED_ENGLISH_WORDS
                ):
                    trimmed_end -= 1
                if trimmed_end - start >= min_window:
                    return text[tokens[start].start() : tokens[trimmed_end - 1].end()]
                return text[tokens[start].start() : tokens[end - 1].end()]
    return ""


def english_leak_segments(text: str) -> list[str]:
    """Detecta segmentos longos que ainda parecem estar em ingles."""
    if not text:
        return []
    flagged: list[str] = []
    embedded_english = _embedded_english_window(text)
    if embedded_english:
        return [embedded_english]
    short_phrase_match = re.search(
        rf"\b(?:{'|'.join(re.escape(phrase) for phrase in sorted(SHORT_ENGLISH_LEAKS))})\b",
        text,
        flags=re.IGNORECASE,
    )
    if short_phrase_match:
        return [short_phrase_match.group(0)]
    mixed_artifact_match = MIXED_ENGLISH_ARTIFACT_RE.search(text)
    if mixed_artifact_match:
        return [mixed_artifact_match.group(0)]
    # Tokens únicos são o último recurso. Quando fazem parte de uma frase
    # inglesa maior, o reparo deve receber a frase/parágrafo completo, não só
    # uma palavra como ``they``.
    single_token_match = re.search(
        rf"\b(?:{'|'.join(re.escape(word) for word in sorted(SINGLE_TOKEN_ENGLISH_LEAKS))})\b",
        text,
        flags=re.IGNORECASE,
    )
    if single_token_match:
        return [single_token_match.group(0)]
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
        portuguese_hits = sum(
            1 for word in normalized if word in PORTUGUESE_ANCHOR_WORDS
        )
        dense_english = (
            len(words) >= 10
            and english_hits >= 5
            and english_hits / max(len(words), 1) >= 0.25
        )
        dominant_english = english_hits >= 4 and english_hits >= portuguese_hits + 2
        if dense_english or dominant_english:
            flagged.append(segment)
    return flagged


def detect_residual_english(text: str) -> tuple[bool, str]:
    """Processamento interno auxiliar."""
    segments = english_leak_segments(text)
    if not segments:
        return False, ""
    preview = re.sub(r"\s+", " ", segments[0]).strip()
    if len(preview) > 80:
        preview = preview[:77].rstrip() + "..."
    return True, f"residual_english:{preview}"
