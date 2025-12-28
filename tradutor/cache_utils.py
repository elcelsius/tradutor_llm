"""
Utilidades de cache por chunk e detecção de colapso.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import re

_CACHE_BASE_DIR: Path = Path("saida")


def set_cache_base_dir(base_dir: str | Path) -> None:
    """Define o diretório base para todos os caches (padrão: saida)."""
    global _CACHE_BASE_DIR
    _CACHE_BASE_DIR = Path(base_dir)


def _cache_dirs() -> dict[str, Path]:
    base = _CACHE_BASE_DIR
    return {
        "translate": base / "cache_traducao",
        "refine": base / "cache_refine",
        "desquebrar": base / "cache_desquebrar",
    }


def chunk_hash(text: str) -> str:
    """Gera hash curto (16 hex) do conteúdo exato do chunk."""
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return h[:16]


def _cache_path(mode: str, h: str) -> Path:
    dirs = _cache_dirs()
    base = dirs.get(mode, _CACHE_BASE_DIR / "cache_misc")
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


def detect_model_collapse(
    text: str,
    original_len: int | None = None,
    mode: str = "translate",
    return_reasons: bool = False,
) -> bool | tuple[bool, list[str], dict]:
    """
    Heurística simples para detectar saída corrompida/colapsada.
    Quando return_reasons=True, retorna (flag, reasons, details).
    """
    reasons: list[str] = []
    details: dict[str, int | float] = {}

    if mode == "translate":
        return (False, reasons, details) if return_reasons else False
    if not text:
        reasons.append("empty")
        flag = True
        return (flag, reasons, details) if return_reasons else flag

    # Repetição de linhas (considera apenas linhas mais longas)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) >= 20]
    details["line_candidates"] = len(lines)
    counts: dict[str, int] = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    if counts:
        details["max_line_repeats"] = max(counts.values())
        if any(c >= 3 for c in counts.values()):
            reasons.append("repeated_lines")

    # Loop de tokens consecutivos (palavra com 3+ letras repetida 8+ vezes)
    token_run = re.search(r"\b(\w{3,})(?:\s+\1){7,}\b", text, flags=re.IGNORECASE)
    if token_run:
        repeated_seq = token_run.group(0)
        details["max_token_run"] = len(re.findall(r"\b\w+\b", repeated_seq))
        reasons.append("repeated_token_run")

    # CJK em excesso
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    total_chars = len(text)
    details["cjk_chars"] = cjk
    details["cjk_ratio"] = cjk / max(total_chars, 1)
    if cjk > 10:
        reasons.append("excess_cjk")

    # Símbolos indevidos
    symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    details["symbol_ratio"] = symbol_count / max(total_chars, 1)
    if ("$$$$" in text) or ("<think>" in text) or ("<analysis>" in text) or details["symbol_ratio"] > 0.35:
        reasons.append("bad_symbols")
    if "###" in text and "TEXTO_TRADUZIDO" not in text and "TEXTO_REFINADO" not in text:
        reasons.append("bad_symbols")

    # Tamanho relativo
    if original_len:
        ratio = len(text) / max(original_len, 1)
        details["ratio_out_in"] = ratio
        if mode == "translate":
            if ratio < 0.7 or ratio > 2.0:
                reasons.append("bad_ratio")
        elif mode == "refine":
            if ratio < 0.5 or ratio > 3.0:
                reasons.append("bad_ratio")
        else:
            if ratio < 0.5 or ratio > 3.0:
                reasons.append("bad_ratio")

    strong = {"empty", "repeated_token_run", "bad_ratio", "bad_symbols", "excess_cjk", "repeated_lines"}
    flag = any(r in strong for r in reasons)
    return (flag, reasons, details) if return_reasons else flag


def clear_cache(mode: str = "all") -> None:
    """
    Remove diretórios de cache respeitando o _CACHE_BASE_DIR atual.
    mode: all | translate | refine | desquebrar
    """
    dirs = _cache_dirs()
    targets: dict[str, Path] = {}
    if mode == "all":
        targets = dirs
    else:
        selected = dirs.get(mode)
        if selected:
            targets[mode] = selected
    for _, path in targets.items():
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
