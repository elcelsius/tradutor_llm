"""
Camada Anti-Alucinação (AAA Shield Mode) para tradução e refine.
Determinística, sem IA, aplicada pós-sanitização.
"""

from __future__ import annotations

import re
from typing import List


def detect_language_anomaly(text: str, mode: str = "refine") -> bool:
    if not text:
        return True
    lower = text.lower()
    cjk_blocks = re.findall(r"[\u4e00-\u9fff]{6,}", text)
    if cjk_blocks:
        return True
    french_es = ["mon ami", "bonjour", "ma ch", "très", "oui", "siempre", "porque", "pero", "esta ", "está "]
    if any(pat in lower for pat in french_es):
        return True
    if mode != "translate":
        english_words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
        if english_words:
            common_en = {"the", "and", "with", "from", "this", "that", "here", "there", "you", "your", "their"}
            en_hits = sum(1 for w in english_words if w.lower() in common_en)
            if en_hits / max(len(english_words), 1) > 0.03:
                return True
    markers = ["as an ai", "here is the refined text", "<think>", "</think>", "assistant:", "user:"]
    if any(pat in lower for pat in markers):
        return True
    return False


def detect_repetition_anomaly(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    counts = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    if any(c >= 3 for c in counts.values()):
        return True
    words = text.split()
    wc = {}
    for w in words:
        wc[w] = wc.get(w, 0) + 1
    if words and max(wc.values()) >= 10:
        return True
    return False


def detect_structure_anomaly(text: str) -> bool:
    lower = text.lower()
    if "### texto_traduzido_inicio".lower() in lower and "### texto_traduzido_fim".lower() not in lower:
        return True
    if "### texto_refinado_inicio".lower() in lower and "### texto_refinado_fim".lower() not in lower:
        return True
    bad_markers = ["<think>", "assistant:", "user:", "===glossario_s", "```", "{", "}", "[", "]"]
    if any(bm in lower for bm in bad_markers):
        return True
    return False


def _extract_entities(text: str) -> List[str]:
    return [m.group() for m in re.finditer(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÄÖÜ][\wÁÉÍÓÚÂÊÔÃÕÄÖÜ-]{2,}\b", text)]


def detect_semantic_drift(orig: str, llm: str) -> bool:
    orig_entities = _extract_entities(orig)
    llm_entities = _extract_entities(llm)
    if not orig_entities:
        return False
    shared = len(set(orig_entities) & set(llm_entities))
    if shared / max(len(orig_entities), 1) < 0.6:
        return True
    if len(llm_entities) > len(orig_entities) + 5:
        return True
    return False


def sanitize_llm_output(llm_raw: str) -> str:
    cleaned = llm_raw
    cleaned = cleaned.replace("Here is the refined text:", "")
    cleaned = cleaned.replace("Texto refinado:", "")
    cleaned = cleaned.replace("Here is the text:", "")
    cleaned = re.sub(r"```.+?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\s{3,}", " ", cleaned)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def anti_hallucination_filter(orig: str, llm_raw: str, cleaned: str, mode: str) -> str:
    """
    Se qualquer anomalia for detectada, retorna orig (fallback).
    modo: "translate" ou "refine".
    """
    # Para tradução EN->PT, não fazemos fallback automático para o original
    # para evitar voltar ao inglês; apenas sanitizamos.
    if mode == "translate":
        return sanitize_llm_output(cleaned)

    safe = sanitize_llm_output(cleaned)
    if detect_language_anomaly(safe, mode=mode):
        return orig
    if detect_repetition_anomaly(safe):
        return orig
    if detect_structure_anomaly(llm_raw):
        return orig
    # Deriva semântica só faz sentido no refine (orig e alvo na mesma língua).
    if mode == "refine" and detect_semantic_drift(orig, safe):
        return orig
    return safe
