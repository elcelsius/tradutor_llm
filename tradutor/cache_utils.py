"""
Utilidades de cache por chunk e detecção de colapso.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import re

CACHE_DIRS = {
    "translate": Path("saida/cache_traducao"),
    "refine": Path("saida/cache_refine"),
}


def chunk_hash(text: str) -> str:
    """Gera hash curto (16 hex) do conteúdo exato do chunk."""
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return h[:16]


def _cache_path(mode: str, h: str) -> Path:
    base = CACHE_DIRS.get(mode, Path("saida/cache_misc"))
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{h}.json"


def cache_exists(mode: str, h: str) -> bool:
    return _cache_path(mode, h).exists()


def load_cache(mode: str, h: str) -> Dict[str, Any]:
    path = _cache_path(mode, h)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(mode: str, h: str, raw_output: str, final_output: str, metadata: Dict[str, Any]) -> None:
    path = _cache_path(mode, h)
    payload = {
        "hash": h,
        "raw_output": raw_output,
        "final_output": final_output,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # falha silenciosa no cache não deve quebrar pipeline
        return


def is_near_duplicate(a: str, b: str, threshold: float = 0.95) -> bool:
    """Checagem simples de similaridade para reuso de chunk."""
    import difflib

    a_norm = " ".join(a.split())
    b_norm = " ".join(b.split())
    if not a_norm or not b_norm:
        return False
    ratio = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= threshold


def detect_model_collapse(text: str, original_len: int | None = None, mode: str = "translate") -> bool:
    """Heurística simples para detectar saída corrompida/colapsada."""
    if not text:
        return True

    # Repetição de linhas
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    counts = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    if any(c >= 3 for c in counts.values()):
        return True

    # Loop de tokens simples (palavra repetida muitas vezes)
    words = re.findall(r"\w+", text.lower())
    wc = {}
    for w in words:
        wc[w] = wc.get(w, 0) + 1
    if words and max(wc.values()) >= 10:
        return True

    # CJK ou francês/espanhol em excesso
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    if cjk > 10:
        return True
    accent = len(re.findall(r"[éèêçôàùáíóúñ]", text.lower()))
    french_words = len(re.findall(r"\b(?:bonjour|mon ami|ma ch[eè]re|oui|non)\b", text.lower()))
    if accent > 30 or french_words >= 2:
        return True

    # Símbolos indevidos
    if ("$$$$" in text) or ("<think>" in text) or ("<analysis>" in text):
        return True
    if "###" in text and "TEXTO_TRADUZIDO" not in text and "TEXTO_REFINADO" not in text:
        return True

    # Tamanho relativo
    if original_len:
        ratio = len(text) / max(original_len, 1)
        if mode == "translate" and ratio < 0.7:
            return True
        if mode == "refine" and ratio < 0.8:
            return True
        if ratio > 2.0:
            return True

    return False
