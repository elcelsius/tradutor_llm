"""
Modo Editor Profissional (opcional) pós-refine.
As rotinas são determinísticas, sem IA, e preservam significado/eventos.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

Change = Dict[str, object]


def _record_change(changes: List[Change], before: str, after: str, line: int, reason: str, mode: str) -> None:
    if before != after:
        changes.append({"before": before, "after": after, "line": line, "reason": reason, "mode": mode})


def editor_lite(text: str) -> Tuple[str, Dict]:
    """Pequenas correções estilísticas sem alterar conteúdo."""
    lines = text.splitlines()
    out: List[str] = []
    changes: List[Change] = []
    dup_word_re = re.compile(r"\b(\w+)\s+\1\b", flags=re.IGNORECASE)
    for idx, ln in enumerate(lines, start=1):
        original = ln
        ln = dup_word_re.sub(r"\1", ln)
        ln = re.sub(r"\s+([,.;!?])", r"\1", ln)
        ln = re.sub(r"\s{2,}", " ", ln)
        # vírgula antes de mas/ou/e quando claramente isolada
        ln = re.sub(r",\s+(mas|ou|e)\b", r" \1", ln, flags=re.IGNORECASE)
        _record_change(changes, original, ln, idx, "lite ajustes leves", "editor-lite")
        out.append(ln)
    return "\n".join(out), {"changes": len(changes), "detail": changes}


def editor_consistency(text: str, memory: Dict | None = None) -> Tuple[str, Dict]:
    """Padroniza capitalização/termos comuns mantendo estilo local."""
    memory = memory or {}
    replacements = {
        r"\bslime dourado\b": "Slime Dourado",
        r"\bdeusa vicius\b": "Deusa Vicius",
        r"\btouka-chan\b": "Touka",
        r"\bmuro de pedra\b": "parede de pedra",
    }
    lines = text.splitlines()
    out: List[str] = []
    changes: List[Change] = []
    for idx, ln in enumerate(lines, start=1):
        original = ln
        for pat, rep in replacements.items():
            ln = re.sub(pat, rep, ln, flags=re.IGNORECASE)
        # tempo verbal simples: se predominância de passado detectada, priorizar "era" sobre "é" em descrições
        if memory.get("past_preference"):
            ln = re.sub(r"\b[eE]ra como se ele é\b", "era como se ele era", ln)
        _record_change(changes, original, ln, idx, "consistency padronização local", "editor-consistency")
        out.append(ln)
    memory["changes"] = memory.get("changes", 0) + len(changes)
    return "\n".join(out), {"changes": len(changes), "detail": changes}


def editor_voice(text: str, character_map: Dict | None = None) -> Tuple[str, Dict]:
    """Harmoniza pontuação/ritmo de falas mantendo voz básica."""
    character_map = character_map or {
        "Touka": {"voz": "seca"},
        "Kirihara": {"voz": "arrogante"},
        "Takeda": {"voz": "bruta"},
    }
    lines = text.splitlines()
    out: List[str] = []
    changes: List[Change] = []
    for idx, ln in enumerate(lines, start=1):
        original = ln
        # suaviza reticências exageradas dentro de falas
        if ln.lstrip().startswith("—"):
            ln = re.sub(r"\.{4,}", "…", ln)
            ln = re.sub(r"\s{2,}", " ", ln)
        _record_change(changes, original, ln, idx, "voice ritmo de fala", "editor-voice")
        out.append(ln)
    return "\n".join(out), {"changes": len(changes), "detail": changes, "character_map": character_map}


def editor_strict(text: str) -> Tuple[str, Dict]:
    """Edição literária forte, mas ainda conservadora quanto a significado."""
    lines = text.splitlines()
    out: List[str] = []
    changes: List[Change] = []
    for idx, ln in enumerate(lines, start=1):
        original = ln
        # elimina literalismos: "tipo," desnecessário
        ln = re.sub(r"\btipo,\s*", "", ln)
        # frases duras: "era como se ele fosse tipo" -> "era como se ele fosse"
        ln = ln.replace("como se ele fosse tipo", "como se ele fosse")
        # comprime repetições triviais
        ln = re.sub(r"\b(muito\s+){2,}", "muito ", ln)
        ln = re.sub(r"\s{2,}", " ", ln)
        ln = re.sub(r"\s+([,.;!?])", r"\1", ln)
        _record_change(changes, original, ln, idx, "strict fluidez", "editor-strict")
        out.append(ln)
    return "\n".join(out), {"changes": len(changes), "detail": changes}


def editor_pipeline(text: str, flags: Dict[str, bool]) -> Tuple[str, List[Change]]:
    """Executa os modos selecionados em sequência e coleta mudanças."""
    changes: List[Change] = []
    current = text
    memory: Dict = {}
    if flags.get("lite"):
        current, info = editor_lite(current)
        changes.extend(info.get("detail", []))
    if flags.get("consistency"):
        current, info = editor_consistency(current, memory)
        changes.extend(info.get("detail", []))
    if flags.get("voice"):
        current, info = editor_voice(current)
        changes.extend(info.get("detail", []))
    if flags.get("strict"):
        current, info = editor_strict(current)
        changes.extend(info.get("detail", []))
    return current, changes
