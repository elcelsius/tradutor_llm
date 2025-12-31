from __future__ import annotations

import re

ELLIPSIS_IN_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]\.\.\.[A-Za-zÀ-ÿ]|[A-Za-zÀ-ÿ]…[A-Za-zÀ-ÿ]")
LOWERCASE_START_RE = re.compile(r"^[a-zà-ÿ].*")
TRUNCATED_ELLIPSIS_RE = re.compile(r"(?:\b[A-Za-zÀ-ÿ]+(?:\.{3}|…)\b|\b(?:\.{3}|…)[A-Za-zÀ-ÿ]+)")


def _has_suspicious_repetition(text: str, min_repeats: int = 3) -> bool:
    """Sinais fortes; precisa de 2 ou mais."""
    signals = 0
    if re.search(r"(.{120,}?)(?:\s+\1){1,}", text, flags=re.DOTALL):
        signals += 1
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) >= 40:
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        if unique_ratio < 0.25:
            signals += 1
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    counts = {}
    for s in sentences:
        counts[s] = counts.get(s, 0) + 1
    if any(c >= min_repeats for c in counts.values()):
        signals += 1
    return signals >= 2


def _has_meta_noise(text: str) -> bool:
    lower = text.lower()
    markers = ["as an ai", "<think>", "</think>", "sou um modelo de linguagem", "analysis:"]
    return any(m in lower for m in markers)


def count_quotes(text: str) -> int:
    return len([ch for ch in text if ch in {'"', "“", "”", "‟", "❝", "❞"}])


def count_quote_lines(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip().startswith(('"', "“", "”")))


def needs_retry(
    input_text: str,
    output_text: str,
    *,
    input_quotes: int | None = None,
    output_quotes: int | None = None,
    input_quote_lines: int | None = None,
    output_quote_lines: int | None = None,
    contamination_detected: bool = False,
    sanitization_ratio: float = 1.0,
) -> tuple[bool, str]:
    iq = input_quotes if input_quotes is not None else count_quotes(input_text)
    oq = output_quotes if output_quotes is not None else count_quotes(output_text)
    if iq >= 4 and oq < iq - 2:
        return True, "omissao_dialogo_quotes"
    iql = input_quote_lines if input_quote_lines is not None else count_quote_lines(input_text)
    oql = output_quote_lines if output_quote_lines is not None else count_quote_lines(output_text)
    if iql >= 2 and oql < max(1, iql - 1):
        return True, "omissao_dialogo_linhas"
    if not output_text or not output_text.strip():
        return True, "output vazio"
    if ELLIPSIS_IN_WORD_RE.search(output_text):
        return True, "ellipsis_in_word"
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", output_text) if p.strip()]
    for p in paragraphs:
        if len(paragraphs) > 1 and len(p) <= 30 and LOWERCASE_START_RE.match(p) and not p.startswith(('"', "“", "”", "-", "—")):
            return True, "lowercased_fragment"
    for p in paragraphs:
        if LOWERCASE_START_RE.match(p) and not p.startswith(('"', "“", "”", "-", "—")):
            return True, "lowercase_narration_start"
    if TRUNCATED_ELLIPSIS_RE.search(output_text) or re.search(r"[A-Za-zÀ-ÿ]{1,6}\.{3}", output_text) or re.search(r"[A-Za-zÀ-ÿ]{1,6}…", output_text):
        return True, "truncated_token_ellipsis"
    input_ellipsis = input_text.count("...") + input_text.count("…")
    output_ellipsis = output_text.count("...") + output_text.count("…")
    if input_ellipsis == 0 and output_ellipsis >= 2:
        ellipsis_ratio = output_ellipsis / max(len(output_text), 1)
        if len(output_text) < 500 or ellipsis_ratio > 0.01:
            return True, "ellipsis_suspect"
    ratio = len(output_text.strip()) / max(len(input_text.strip()), 1)
    if ratio < 0.6:
        return True, "output truncado (ratio < 0.6)"
    if _has_suspicious_repetition(output_text):
        return True, "repeticao suspeita"
    if _has_meta_noise(output_text):
        return True, "meta noise detectado"
    if contamination_detected and sanitization_ratio < 0.95:
        return True, "sanitizacao_agressiva"
    return False, ""
