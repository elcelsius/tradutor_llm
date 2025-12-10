from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

GlossaryEntry = Dict[str, Any]
GlossaryIndex = Dict[str, GlossaryEntry]
GlossaryPtIndex = Dict[str, GlossaryEntry]

GLOSSARIO_SUGERIDO_INICIO = "===GLOSSARIO_SUGERIDO_INICIO==="
GLOSSARIO_SUGERIDO_FIM = "===GLOSSARIO_SUGERIDO_FIM==="
DEFAULT_GLOSSARY_PROMPT_LIMIT = 100


def normalize_key(key: str) -> str:
    """Normaliza a chave do glossário para comparação/índice."""
    return key.strip().lower()


def normalize_value(value: str) -> str:
    """Normaliza textos (key/pt) para comparação insensível a caixa/espaços."""
    return value.strip().lower()


def _is_valid_dynamic_term(candidate: str, logger: logging.Logger) -> bool:
    """Aplica filtros de sanidade para evitar termos dinâmicos descritivos demais."""
    cand = candidate.strip()
    if not cand:
        return False
    if len(cand) > 80:
        logger.info("Ignorando termo dinâmico muito longo: %r", cand)
        return False
    if len(cand.split()) > 6:
        logger.info("Ignorando termo dinâmico com muitas palavras: %r", cand)
        return False
    lowered = f" {cand.lower()} "
    if " que " in lowered or " uma " in lowered or " um " in lowered:
        logger.info("Ignorando termo dinâmico com padrão de frase: %r", cand)
        return False
    return True


def _build_index(terms: List[GlossaryEntry]) -> GlossaryIndex:
    return {normalize_key(str(term.get("key", ""))): term for term in terms if str(term.get("key", "")).strip()}


def _build_manual_pt_index(terms: List[GlossaryEntry]) -> GlossaryPtIndex:
    """Índice auxiliar por campo pt (normalizado) para evitar duplicar conceitos no dinâmico."""
    idx: GlossaryPtIndex = {}
    for term in terms:
        pt_raw = str(term.get("pt", "")).strip()
        if not pt_raw:
            continue
        pt_norm = normalize_value(pt_raw)
        if pt_norm and pt_norm not in idx:
            idx[pt_norm] = term
    return idx


def _merge_indexes(manual_index: GlossaryIndex, dynamic_index: GlossaryIndex) -> GlossaryIndex:
    merged = dict(manual_index)
    for key, entry in dynamic_index.items():
        if key not in merged:
            merged[key] = entry
    return merged


def _load_terms(path: Path, source: str, logger: logging.Logger) -> List[GlossaryEntry]:
    if not path.exists():
        logger.info("Glossário %s não encontrado em %s; prosseguindo com vazio.", source, path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - leitura/parse
        logger.warning("Falha ao ler glossário %s em %s: %s", source, path, exc)
        return []
    raw_terms = data.get("terms") if isinstance(data, dict) else None
    if not isinstance(raw_terms, list):
        logger.warning("Formato inesperado no glossário %s em %s; usando lista vazia.", source, path)
        return []

    terms: List[GlossaryEntry] = []
    for entry in raw_terms:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key", "")).strip()
        pt = str(entry.get("pt", "")).strip()
        if not key or not pt:
            continue
        normalized: GlossaryEntry = {
            "key": key,
            "pt": pt,
            "category": entry.get("category"),
            "notes": entry.get("notes"),
            "source": "manual" if source == "manual" else "dynamic",
            "locked": bool(entry.get("locked", source == "manual")),
        }
        terms.append(normalized)
    logger.info("Glossário %s carregado: %d termos.", source, len(terms))
    return terms


def _load_terms_from_dir(dir_path: Path, logger: logging.Logger) -> List[GlossaryEntry]:
    """
    Carrega todos os arquivos *.json de um diretório como glossário manual agregado.
    Mantém apenas a primeira ocorrência de cada key para evitar sobrescritas acidentais.
    """
    if not dir_path.exists():
        logger.info("Diretório de glossário não encontrado em %s; prosseguindo sem auto-glossary.", dir_path)
        return []
    if not dir_path.is_dir():
        logger.warning("Caminho de auto-glossary não é um diretório: %s", dir_path)
        return []

    aggregated: List[GlossaryEntry] = []
    seen: set[str] = set()
    for file in sorted(dir_path.glob("*.json")):
        terms = _load_terms(file, "manual", logger)
        for term in terms:
            key_norm = normalize_key(str(term.get("key", "")))
            if not key_norm or key_norm in seen:
                continue
            seen.add(key_norm)
            aggregated.append(term)
    logger.info("Auto-glossary: %d termos carregados de %s", len(aggregated), dir_path)
    return aggregated


@dataclass
class GlossaryState:
    manual_terms: List[GlossaryEntry]
    dynamic_terms: List[GlossaryEntry]
    manual_index: GlossaryIndex
    dynamic_index: GlossaryIndex
    combined_index: GlossaryIndex
    dynamic_path: Path | None
    manual_pt_index: GlossaryPtIndex

    def refresh_combined(self) -> None:
        """Recalcula índices combinados a partir das listas atuais."""
        self.manual_index = _build_index(self.manual_terms)
        self.dynamic_index = _build_index(self.dynamic_terms)
        self.manual_pt_index = _build_manual_pt_index(self.manual_terms)
        self.combined_index = _merge_indexes(self.manual_index, self.dynamic_index)


def build_glossary_state(
    manual_path: Path | None,
    dynamic_path: Path | None,
    logger: logging.Logger,
    manual_dir: Path | None = None,
) -> GlossaryState | None:
    """Carrega glossários manual/dinâmico (e auto-glossary opcional) e retorna estado consolidado."""
    if manual_path is None and dynamic_path is None and manual_dir is None:
        return None

    manual_terms: List[GlossaryEntry] = []
    if manual_dir:
        manual_terms.extend(_load_terms_from_dir(manual_dir, logger))
    if manual_path:
        manual_terms.extend(_load_terms(manual_path, "manual", logger))
    dynamic_terms = _load_terms(dynamic_path, "dynamic", logger) if dynamic_path else []

    state = GlossaryState(
        manual_terms=manual_terms,
        dynamic_terms=dynamic_terms,
        manual_index=_build_index(manual_terms),
        dynamic_index=_build_index(dynamic_terms),
        combined_index={},
        dynamic_path=dynamic_path,
        manual_pt_index=_build_manual_pt_index(manual_terms),
    )
    state.combined_index = _merge_indexes(state.manual_index, state.dynamic_index)
    return state


def format_glossary_for_prompt(combined_index: GlossaryIndex, limit: int = DEFAULT_GLOSSARY_PROMPT_LIMIT) -> str:
    """Gera bloco de texto para o prompt a partir do glossário combinado."""
    if not combined_index:
        return ""
    entries = sorted(combined_index.values(), key=lambda e: normalize_key(str(e.get("key", ""))))[:limit]
    lines = ["GLOSSÁRIO CANÔNICO (use SEMPRE estas traduções):"]
    for entry in entries:
        key = str(entry.get("key", "")).strip()
        pt = str(entry.get("pt", "")).strip()
        if not key or not pt:
            continue
        category = entry.get("category")
        notes = entry.get("notes")
        line = f"- {key} -> {pt}"
        if category:
            line += f" ({category})"
        if notes:
            line += f" | {notes}"
        lines.append(line)
    return "\n".join(lines)


def split_refined_and_suggestions(text: str) -> Tuple[str, str | None]:
    """
    Separa texto refinado e bloco de glossário sugerido pelos delimitadores.
    Retorna (texto_refinado, bloco_ou_none).
    """
    start = text.find(GLOSSARIO_SUGERIDO_INICIO)
    end = text.find(GLOSSARIO_SUGERIDO_FIM)
    if start == -1 or end == -1 or end < start:
        return text.strip(), None
    refined = text[:start].strip()
    block = text[start + len(GLOSSARIO_SUGERIDO_INICIO) : end].strip()
    return refined, block


def parse_glossary_suggestions(block: str) -> List[GlossaryEntry]:
    """
    Converte bloco textual de sugestões em lista de entradas.
    Formato esperado:
        key: termo
        pt: tradução
        category: opcional
        notes: opcional
        ---
    """
    if not block:
        return []
    suggestions: List[GlossaryEntry] = []
    current: GlossaryEntry = {}

    def flush_current() -> None:
        if current.get("key") and current.get("pt"):
            suggestions.append(
                {
                    "key": str(current["key"]).strip(),
                    "pt": str(current["pt"]).strip(),
                    "category": current.get("category"),
                    "notes": current.get("notes"),
                }
            )

    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "---":
            flush_current()
            current = {}
            continue
        if ":" not in line:
            continue
        field, value = line.split(":", 1)
        field = field.strip().lower()
        value = value.strip()
        if field in {"key", "pt", "category", "notes"}:
            current[field] = value
    flush_current()
    return suggestions


def apply_suggestions_to_state(
    state: GlossaryState,
    suggestions: List[GlossaryEntry],
    logger: logging.Logger,
) -> bool:
    """
    Aplica sugestões no glossário dinâmico respeitando prioridade do manual e flags locked.
    Retorna True se houve mudança no glossário dinâmico.
    """
    if not suggestions:
        return False

    changed = False
    for entry in suggestions:
        key_raw = str(entry.get("key", "")).strip()
        pt = str(entry.get("pt", "")).strip()
        # Para o refinador, tratamos key/pt como o mesmo rótulo em PT-BR
        term_pt = pt or key_raw
        if not key_raw or not pt or not term_pt.strip():
            continue
        key_norm = normalize_key(term_pt)
        category = entry.get("category")
        notes = entry.get("notes")

        pt_norm = normalize_value(pt) if pt else ""
        if pt_norm and pt_norm in state.manual_pt_index:
            logger.debug("Ignorando sugestão de glossário para '%s' (pt já definido no manual).", key_raw)
            continue

        if key_norm in state.manual_index:
            logger.debug("Ignorando sugestão de glossário para '%s' (definido no manual).", key_raw)
            continue

        existing = state.dynamic_index.get(key_norm)
        if existing:
            if existing.get("locked"):
                logger.debug("Entrada dinâmica '%s' está bloqueada; não será alterada.", existing.get("key"))
                continue
            updated = False
            if term_pt and term_pt != existing.get("pt"):
                existing["pt"] = term_pt
                existing["key"] = term_pt
                updated = True
            if category and category != existing.get("category"):
                existing["category"] = category
                updated = True
            if notes and notes != existing.get("notes"):
                existing["notes"] = notes
                updated = True
            if updated:
                changed = True
                logger.info("Glossário dinâmico atualizado para '%s' -> %s", key_raw, pt)
            continue

        candidate = term_pt.strip()
        if not _is_valid_dynamic_term(candidate, logger):
            continue

        new_entry: GlossaryEntry = {
            "key": candidate,
            "pt": candidate,
            "category": category if category else None,
            "notes": notes if notes else None,
            "source": "dynamic",
            "locked": False,
        }
        state.dynamic_terms.append(new_entry)
        state.dynamic_index[key_norm] = new_entry
        changed = True
        logger.info("Nova entrada adicionada ao glossário dinâmico: %s -> %s", key_raw, pt)

    if changed:
        state.refresh_combined()
    return changed


def save_dynamic_glossary(state: GlossaryState, logger: logging.Logger) -> None:
    """Grava o glossário dinâmico no caminho configurado."""
    if state.dynamic_path is None:
        return
    state.dynamic_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_terms = sorted(state.dynamic_terms, key=lambda t: normalize_key(str(t.get("key", ""))))
    payload = {"terms": sorted_terms}
    try:
        state.dynamic_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Glossário dinâmico salvo em %s (termos: %d).", state.dynamic_path, len(sorted_terms))
    except Exception as exc:  # pragma: no cover - I/O edge case
        logger.warning("Falha ao salvar glossário dinâmico em %s: %s", state.dynamic_path, exc)
