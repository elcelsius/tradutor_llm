"""
Refinamento capítulo a capítulo de arquivos Markdown.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable

from .desquebrar import normalize_md_paragraphs

from .config import AppConfig
from .glossary_utils import (
    DEFAULT_GLOSSARY_PROMPT_LIMIT,
    GLOSSARIO_SUGERIDO_FIM,
    GLOSSARIO_SUGERIDO_INICIO,
    GlossaryState,
    apply_suggestions_to_state,
    build_glossary_state,
    format_glossary_for_prompt,
    parse_glossary_suggestions,
    save_dynamic_glossary,
    split_refined_and_suggestions,
)
from .llm_backend import LLMBackend
from .preprocess import chunk_for_refine, paragraphs_from_text
from .sanitizer import sanitize_refine_output
from .utils import ensure_dir, read_text, timed, write_text
from .cache_utils import (
    cache_exists,
    chunk_hash,
    detect_model_collapse,
    load_cache,
    save_cache,
    is_near_duplicate,
)
from .advanced_preprocess import clean_text as advanced_clean
from .anti_hallucination import anti_hallucination_filter
from .cleanup import cleanup_before_refine, detect_obvious_dupes, detect_glued_dialogues


@dataclass
class RefineStats:
    total_blocks: int = 0
    success_blocks: int = 0
    error_blocks: int = 0


@dataclass
class RefineProgress:
    total_blocks: int
    refined_blocks: set[int]
    error_blocks: set[int]
    chunk_outputs: Dict[int, str]
    progress_path: Path | None


_CURRENT_STATS: RefineStats | None = None
_CURRENT_PROGRESS: RefineProgress | None = None
_GLOBAL_BLOCK_INDEX: int = 0


@contextmanager
def processing_context(stats: RefineStats, progress: RefineProgress | None):
    global _CURRENT_STATS, _CURRENT_PROGRESS, _GLOBAL_BLOCK_INDEX
    prev_stats = _CURRENT_STATS
    prev_progress = _CURRENT_PROGRESS
    prev_counter = _GLOBAL_BLOCK_INDEX
    _CURRENT_STATS = stats
    _CURRENT_PROGRESS = progress
    _GLOBAL_BLOCK_INDEX = 0
    try:
        yield
    finally:
        _CURRENT_STATS = prev_stats
        _CURRENT_PROGRESS = prev_progress
        _GLOBAL_BLOCK_INDEX = prev_counter


def _next_block_index() -> int:
    global _GLOBAL_BLOCK_INDEX
    _GLOBAL_BLOCK_INDEX += 1
    return _GLOBAL_BLOCK_INDEX


def has_suspicious_repetition(text: str, min_repeats: int = 3) -> bool:
    """
    Retorna True se o texto contiver linhas repetidas muitas vezes (possível loop do LLM).
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return False
    counts: Dict[str, int] = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    return any(c >= min_repeats for c in counts.values())


def has_meta_noise(text: str) -> bool:
    """Detecta meta-texto óbvio que não deve aparecer na saída final."""
    lower = text.lower()
    markers = [
        "as an ai",
        "as a language model",
        "sou um modelo de linguagem",
        "como um modelo de linguagem",
        "<think>",
        "</think>",
    ]
    return any(m in lower for m in markers)


def save_refine_debug_files(
    output_dir: Path,
    section_index: int,
    chunk_index: int,
    original_text: str,
    llm_raw: str,
    final_text: str,
    logger: logging.Logger,
) -> None:
    """Salva arquivos de debug para inspeção de refine por chunk."""
    ensure_dir(output_dir)
    base = f"sec{section_index:03d}_chunk{chunk_index:03d}"

    def _write(name: str, content: str) -> None:
        path = output_dir / f"{base}_{name}.txt"
        path.write_text(content, encoding="utf-8")

    _write("original", original_text)
    _write("llm_raw", llm_raw)
    _write("final", final_text)
    logger.info("Debug refine salvo: %s_* em %s", base, output_dir)


def _write_progress(progress: RefineProgress | None, logger: logging.Logger) -> None:
    if progress is None or progress.progress_path is None:
        return
    data = {
        "total_blocks": progress.total_blocks,
        "refined_blocks": sorted(progress.refined_blocks),
        "error_blocks": sorted(progress.error_blocks),
        "timestamp": datetime.now().isoformat(),
        "chunks": {str(idx): text for idx, text in progress.chunk_outputs.items()},
    }
    try:
        progress.progress_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - I/O edge case
        logger.warning("Falha ao gravar manifesto de refine em %s: %s", progress.progress_path, exc)


def _prepare_progress(
    progress_path: Path,
    resume_manifest: dict | None,
    total_blocks: int,
    logger: logging.Logger,
) -> RefineProgress:
    refined: set[int] = set()
    errored: set[int] = set()
    chunk_outputs: Dict[int, str] = {}
    manifest_total = None

    data = resume_manifest
    if isinstance(data, dict):
        manifest_total = data.get("total_blocks")
        if isinstance(manifest_total, int) and manifest_total != total_blocks:
            logger.warning(
                "Manifesto indica %d blocos, mas chunking atual gerou %d; usando chunking atual.",
                manifest_total,
                total_blocks,
            )
        raw_chunks = data.get("chunks") or {}
        if isinstance(raw_chunks, dict):
            for key, val in raw_chunks.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if isinstance(val, str):
                    chunk_outputs[idx] = val

        raw_refined = data.get("refined_blocks") or []
        for idx in raw_refined:
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            if idx_int in chunk_outputs:
                refined.add(idx_int)
            else:
                logger.warning(
                    "Manifesto marca bloco %s como refinado, mas não há conteúdo salvo; refinando novamente.",
                    idx_int,
                )

        raw_error = data.get("error_blocks") or []
        for idx in raw_error:
            try:
                errored.add(int(idx))
            except (TypeError, ValueError):
                continue

    return RefineProgress(
        total_blocks=total_blocks,
        refined_blocks=refined,
        error_blocks=errored,
        chunk_outputs=chunk_outputs,
        progress_path=progress_path,
    )


def build_refine_prompt(section: str, glossary_enabled: bool = False, glossary_block: str | None = None) -> str:
    glossary_section = ""
    if glossary_enabled and glossary_block:
        glossary_section = f"\nUse como referencia (sem adicionar explicacoes) o glossario a seguir:\n{glossary_block}\n"

    return f"""
Você é um EDITOR PROFISSIONAL DE LIGHT NOVELS, responsável por transformar um texto traduzido para o português brasileiro em uma versão natural, fluida, coerente, com tom literário e qualidade de publicação.
Não altere absolutamente nada da história, dos eventos, das falas, da linha do tempo ou do conteúdo original. Apenas melhore a escrita.

OBJETIVOS DO EDITOR:

1. Naturalizar o português brasileiro, transformando frases literais em frases fluidas, claras e naturais.
2. Corrigir gagueiras mal traduzidas, ajustando para formas naturais em PT-BR:

   * "H-he..." → "E-ele..." ou apenas "Ele..."
3. Corrigir possessivos do inglês:

   * "Kirihara's grupo" → "o grupo do Kirihara"
   * "Agit's blade" → "a lâmina do Agit"
4. Adaptar humor e trocadilhos para PT-BR mantendo o espírito da piada:

   * "Four Holey Smell-ders" → "Os Quatro Furados Fedorentos"
   * "Captain Mammaries" → "Capitã Peituda"
5. Remover ruídos de OCR/PDF como aspas triplas \"\"\" e caracteres soltos.
6. Melhorar ritmo, coerência e pontuação de diálogos.
7. Remover calques literais:

   * "O que é com essa atitude de superioridade?!" → "Que atitude é essa, todo se achando?!"
8. Padronizar fluidez narrativa.
9. Ajustar repetições acidentais.
10. Manter consistência de gênero/narrador (masculino/feminino) conforme o original; não inverter narrador masculino.

PROIBIÇÕES ABSOLUTAS:

* Não resumir.
* Não cortar falas.
* Não alterar eventos.
* Não adicionar conteúdo.
* Não mudar tom ou personalidade dos personagens.
* Não reorganizar parágrafos.

FORMATO DE SAÍDA:
Retorne apenas:

### TEXTO_REFINADO_INICIO

<texto refinado>
### TEXTO_REFINADO_FIM

Nada antes ou depois dos marcadores.

{glossary_section}Texto para revisao (PT-BR):
\"\"\"{section}\"\"\"\n"""


def split_markdown_sections(md_text: str) -> List[Tuple[str, str]]:
    """
    Divide o Markdown em seções por headings `##`.

    Retorna lista de tuplas (título, corpo). Se não houver headings,
    retorna uma única seção com título vazio.
    """
    pattern = re.compile(r"^##\s+.+$", flags=re.MULTILINE)
    matches = list(pattern.finditer(md_text))
    sections: List[Tuple[str, str]] = []

    if not matches:
        return [("", md_text.strip())]

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md_text)
        title = match.group().strip()
        body = md_text[start:end].strip()
        sections.append((title, body))
    return sections


def refine_section(
    title: str,
    body: str,
    backend: LLMBackend,
    cfg: AppConfig,
    logger: logging.Logger,
    index: int,
    total: int,
    glossary_state: GlossaryState | None = None,
    glossary_prompt_limit: int = DEFAULT_GLOSSARY_PROMPT_LIMIT,
    debug_refine: bool = False,
    metrics: dict | None = None,
    seen_chunks: list | None = None,
    debug_writer: Callable[[dict], None] | None = None,
) -> str:
    if metrics is None:
        metrics = {}
    if seen_chunks is None:
        seen_chunks = []
    paragraphs = paragraphs_from_text(body)
    chunks = chunk_for_refine(paragraphs, max_chars=cfg.refine_chunk_chars, logger=logger)
    logger.info("Refinando seção %s (%d chunks)", title or f"#{index}", len(chunks))
    refined_parts: List[str] = []
    stats = _CURRENT_STATS
    progress = _CURRENT_PROGRESS
    glossary_block = None

    for c_idx, chunk in enumerate(chunks, start=1):
        block_idx = _next_block_index()
        if stats:
            stats.total_blocks += 1
        guard_mode = getattr(cfg, "refine_guardrails", "strict")
        h = chunk_hash(chunk)
        block_metrics = metrics.setdefault("block_metrics", [])

        def record_block(final_text: str, *, used_fallback: bool = False, from_cache: bool = False, from_duplicate: bool = False, collapse: bool = False) -> None:
            ratio = (len(final_text.strip()) / max(len(chunk.strip()), 1)) if chunk.strip() else 0.0
            block_metrics.append(
                {
                    "block_index": block_idx,
                    "chars_in": len(chunk),
                    "chars_out": len(final_text),
                    "ratio_out_in": round(ratio, 3),
                    "used_fallback": used_fallback,
                    "guardrails_mode": guard_mode,
                    "suspicious_repetition": has_suspicious_repetition(final_text),
                    "from_cache": from_cache,
                    "from_duplicate": from_duplicate,
                    "collapse_detected": collapse,
                }
            )
        llm_raw: str | None = None
        for prev_chunk, prev_final in seen_chunks:
            if is_near_duplicate(prev_chunk, chunk):
                logger.info("Chunk ref-%d/%d-%d/%d marcado como duplicado; reuso habilitado.", index, total, c_idx, len(chunks))
                refined_parts.append(prev_final)
                metrics["duplicates"] = metrics.get("duplicates", 0) + 1
                if stats:
                    stats.success_blocks += 1
                if progress:
                    progress.refined_blocks.add(block_idx)
                    progress.error_blocks.discard(block_idx)
                    progress.chunk_outputs[block_idx] = prev_final
                _write_progress(progress, logger)
                record_block(prev_final, from_duplicate=True)
                if debug_writer:
                    debug_writer(
                        {
                            "para_index": block_idx,
                            "original_text": chunk,
                            "original_chars": len(chunk),
                            "refined_text": prev_final,
                            "refined_chars": len(prev_final),
                            "llm_raw_output": None,
                            "sanitizer_report": None,
                        }
                    )
                continue
        if cache_exists("refine", h):
            data = load_cache("refine", h)
            if not _is_cache_compatible(data):
                logger.debug("Cache de refine ignorado: assinatura diferente de backend/model/num_predict.")
            else:
                cached = data.get("final_output")
                if cached:
                    logger.info("Reusando cache de refine para bloco ref-%d/%d-%d/%d", index, total, c_idx, len(chunks))
                    refined_parts.append(cached)
                    metrics["cache_hits"] = metrics.get("cache_hits", 0) + 1
                    if stats:
                        stats.success_blocks += 1
                    if progress:
                        progress.refined_blocks.add(block_idx)
                        progress.error_blocks.discard(block_idx)
                        progress.chunk_outputs[block_idx] = cached
                    _write_progress(progress, logger)
                    record_block(cached, from_cache=True)
                    if debug_writer:
                        debug_writer(
                            {
                                "para_index": block_idx,
                                "original_text": chunk,
                                "original_chars": len(chunk),
                                "refined_text": cached,
                                "refined_chars": len(cached),
                                "llm_raw_output": None,
                                "sanitizer_report": None,
                            }
                        )
                    continue
        if progress and block_idx in progress.refined_blocks and block_idx in progress.chunk_outputs:
            logger.info("Reusando refinamento salvo para bloco ref-%d/%d-%d/%d", index, total, c_idx, len(chunks))
            refined_parts.append(progress.chunk_outputs[block_idx])
            if stats:
                stats.success_blocks += 1
            _write_progress(progress, logger)
            record_block(progress.chunk_outputs[block_idx])
            if debug_writer:
                reused = progress.chunk_outputs[block_idx]
                debug_writer(
                    {
                        "para_index": block_idx,
                        "original_text": chunk,
                        "original_chars": len(chunk),
                        "refined_text": reused,
                        "refined_chars": len(reused),
                        "llm_raw_output": None,
                        "sanitizer_report": None,
                    }
                )
            continue
        if glossary_state:
            glossary_block = format_glossary_for_prompt(
                glossary_state.combined_index,
                glossary_prompt_limit,
            )
        prompt = build_refine_prompt(
            chunk,
            glossary_enabled=bool(glossary_state),
            glossary_block=glossary_block,
        )
        logger.debug("Refinando seção com %d caracteres...", len(chunk))
        try:
            llm_raw, response_text = _call_with_retry(
                backend=backend,
                prompt=prompt,
                cfg=cfg,
                logger=logger,
                label=f"ref-{index}/{total}-{c_idx}/{len(chunks)}",
                max_retries=1,
            )
            refined_candidate = response_text
            if glossary_state:
                refined_candidate, suggestion_block = split_refined_and_suggestions(llm_raw)
                suggestions = parse_glossary_suggestions(suggestion_block or "")
                if suggestions:
                    updated = apply_suggestions_to_state(glossary_state, suggestions, logger)
                    if updated:
                        save_dynamic_glossary(glossary_state, logger)
                        glossary_block = format_glossary_for_prompt(
                            glossary_state.combined_index,
                            glossary_prompt_limit,
                        )

            collapse_flag = False
            used_fallback = False
            if guard_mode == "off":
                refined_text = refined_candidate
                if not refined_text.strip() or has_meta_noise(refined_text) or has_meta_noise(llm_raw or ""):
                    used_fallback = True
                elif detect_model_collapse(refined_text, original_len=len(chunk), mode="refine"):
                    collapse_flag = True
                    used_fallback = True
            elif guard_mode == "relaxed":
                filtered_text = anti_hallucination_filter(orig=chunk, llm_raw=llm_raw, cleaned=refined_candidate, mode="refine")
                if filtered_text == chunk and refined_candidate.strip() and refined_candidate.strip() != chunk.strip():
                    refined_text = refined_candidate
                else:
                    refined_text = filtered_text
                severe_issue = False
                if not refined_text.strip():
                    severe_issue = True
                elif has_meta_noise(refined_text) or has_meta_noise(llm_raw or ""):
                    severe_issue = True
                elif detect_model_collapse(refined_text, original_len=len(chunk), mode="refine"):
                    collapse_flag = True
                    severe_issue = True
                if severe_issue:
                    used_fallback = True
            else:  # strict
                refined_text = anti_hallucination_filter(orig=chunk, llm_raw=llm_raw, cleaned=refined_candidate, mode="refine")
                if not refined_text.strip():
                    used_fallback = True
                elif detect_model_collapse(refined_text, original_len=len(chunk), mode="refine"):
                    collapse_flag = True
                    used_fallback = True

            if used_fallback:
                refined_text = chunk
                metrics["fallbacks"] = metrics.get("fallbacks", 0) + 1
                if collapse_flag:
                    metrics["collapse"] = metrics.get("collapse", 0) + 1
            else:
                ratio = len(refined_text.strip()) / max(len(chunk.strip()), 1)
                if ratio < 0.8 or ratio > 1.8:
                    logger.info(
                        "Refine: divergência de tamanho aceita (mode=%s, ratio=%.2f) no chunk %d/%d da seção %s.",
                        guard_mode,
                        ratio,
                        c_idx,
                        len(chunks),
                        title or f"#{index}",
                    )
                if has_suspicious_repetition(refined_text):
                    logger.warning(
                        "Refinador devolveu texto com repetição suspeita; aceitando (mode=%s) no chunk %d/%d da seção %s.",
                        guard_mode,
                        c_idx,
                        len(chunks),
                        title or f"#{index}",
                    )
            # Debug opcional: salva até os 5 primeiros chunks
            if debug_refine and block_idx <= 5:
                debug_dir = cfg.output_dir / "debug_refine"
                save_refine_debug_files(
                    output_dir=debug_dir,
                    section_index=index,
                    chunk_index=c_idx,
                    original_text=chunk,
                    llm_raw=llm_raw,
                    final_text=refined_text,
                    logger=logger,
                )
            logger.debug("Seção refinada com %d caracteres.", len(refined_text))
            refined_parts.append(refined_text)
            if stats:
                stats.success_blocks += 1
            if progress:
                progress.refined_blocks.add(block_idx)
                progress.error_blocks.discard(block_idx)
                progress.chunk_outputs[block_idx] = refined_text
            seen_chunks.append((chunk, refined_text))
            record_block(refined_text, used_fallback=used_fallback, collapse=collapse_flag)
            save_cache(
                "refine",
                h,
                raw_output=llm_raw,
                final_output=refined_text,
                metadata={
                    "chunk_index": c_idx,
                    "section_index": index,
                    "mode": "refine",
                    "backend": getattr(backend, "backend", None),
                    "model": getattr(backend, "model", None),
                    "num_predict": getattr(backend, "num_predict", None),
                    "temperature": getattr(backend, "temperature", None),
                    "repeat_penalty": getattr(backend, "repeat_penalty", None),
                    "guardrails": getattr(cfg, "refine_guardrails", None),
                },
            )
            if debug_writer:
                debug_writer(
                    {
                        "para_index": block_idx,
                        "original_text": chunk,
                        "original_chars": len(chunk),
                        "refined_text": refined_text,
                        "refined_chars": len(refined_text),
                        "llm_raw_output": llm_raw,
                        "sanitizer_report": None,
                    }
                )
        except RuntimeError as exc:
            logger.warning(
                "Chunk ref-%d/%d-%d/%d falhou; usando texto original. Erro: %s",
                index,
                total,
                c_idx,
                len(chunks),
                exc,
            )
            refined_parts.append(chunk)
            if stats:
                stats.error_blocks += 1
            if progress:
                progress.error_blocks.add(block_idx)
                progress.chunk_outputs[block_idx] = chunk
            metrics["fallbacks"] = metrics.get("fallbacks", 0) + 1
            record_block(chunk, used_fallback=True)
            if debug_writer:
                debug_writer(
                    {
                        "para_index": block_idx,
                        "original_text": chunk,
                        "original_chars": len(chunk),
                        "refined_text": chunk,
                        "refined_chars": len(chunk),
                        "llm_raw_output": llm_raw,
                        "sanitizer_report": None,
                    }
                )
        finally:
            _write_progress(progress, logger)

    refined_section = "\n\n".join(refined_parts).strip()
    if title:
        return f"{title}\n\n{refined_section}"
    return refined_section


def refine_markdown_file(
    input_path: Path,
    output_path: Path,
    backend: LLMBackend,
    cfg: AppConfig,
    logger: logging.Logger,
    progress_path: Path | None = None,
    resume_manifest: dict | None = None,
    normalize_paragraphs: bool = False,
    glossary_state: GlossaryState | None = None,
    glossary_prompt_limit: int = DEFAULT_GLOSSARY_PROMPT_LIMIT,
    debug_refine: bool = False,
    parallel_workers: int = 1,
    preprocess_advanced: bool = False,
    debug_chunks: bool = False,
    cleanup_mode: str = "off",
) -> None:
    raw_md = read_text(input_path)
    md_text = raw_md
    if preprocess_advanced:
        md_text = advanced_clean(md_text)
    if normalize_paragraphs:
        md_text = normalize_md_paragraphs(md_text)
    cleanup_preview_hash_before = chunk_hash(md_text)

    cleanup_applied = False
    cleanup_stats: dict = {}
    cleanup_mode = cleanup_mode if cleanup_mode in ("off", "auto", "on") else "off"
    trigger_cleanup = False
    if cleanup_mode == "on":
        trigger_cleanup = True
    elif cleanup_mode == "auto":
        if detect_obvious_dupes(raw_md) or detect_glued_dialogues(raw_md):
            trigger_cleanup = True
        elif getattr(cfg, "refine_guardrails", "strict") == "strict" and (
            detect_obvious_dupes(raw_md) or detect_glued_dialogues(raw_md)
        ):
            trigger_cleanup = True
    if trigger_cleanup:
        md_text, cleanup_stats = cleanup_before_refine(md_text)
        cleanup_applied = True
        pre_refine_path = output_path.with_name(f"{output_path.stem}_pre_refine_cleanup.md")
        pre_refine_path.write_text(md_text, encoding="utf-8")
    cleanup_preview_hash_after = chunk_hash(md_text)

    doc_hash = chunk_hash(md_text)
    sections = split_markdown_sections(md_text)
    logger.info("Arquivo %s: %d seções detectadas", input_path.name, len(sections))
    logger.info("Refine guardrails mode: %s", getattr(cfg, "refine_guardrails", "strict"))
    stats = RefineStats()
    metrics: dict[str, int | list | dict | bool | str] = {
        "cache_hits": 0,
        "fallbacks": 0,
        "collapse": 0,
        "duplicates": 0,
        "block_metrics": [],
    }
    metrics["cleanup_mode"] = cleanup_mode
    metrics["cleanup_applied"] = cleanup_applied
    metrics["cleanup_stats"] = cleanup_stats
    metrics["cleanup_preview_hash_before"] = cleanup_preview_hash_before
    metrics["cleanup_preview_hash_after"] = cleanup_preview_hash_after
    seen_chunks: list[tuple[str, str]] = []
    cache_signature = {
        "backend": getattr(backend, "backend", None),
        "model": getattr(backend, "model", None),
        "num_predict": getattr(backend, "num_predict", None),
        "temperature": getattr(backend, "temperature", None),
        "repeat_penalty": getattr(backend, "repeat_penalty", None),
        "guardrails": getattr(cfg, "refine_guardrails", None),
    }

    def _is_cache_compatible(data: dict) -> bool:
        meta = data.get("metadata")
        if not isinstance(meta, dict):
            return False
        return all(meta.get(k) == v for k, v in cache_signature.items())

    # Pré-computa total de blocos para progress
    total_blocks = 0
    max_refine_chunk_len = 0
    for _, body in sections:
        paragraphs = paragraphs_from_text(body)
        chunks = chunk_for_refine(paragraphs, max_chars=cfg.refine_chunk_chars, logger=logger)
        total_blocks += len(chunks)
        if chunks:
            max_refine_chunk_len = max(max_refine_chunk_len, max(len(c) for c in chunks))
    metrics["effective_refine_chunk_chars"] = cfg.refine_chunk_chars
    metrics["max_chunk_chars_observed"] = max_refine_chunk_len

    if progress_path is None:
        progress_path = output_path.with_name(f"{output_path.stem}_progress.json")

    state_path = output_path.parent / "state_refine.json"
    debug_file = None
    debug_file_path: Path | None = None

    try:
        state_payload = {
            "input_file": str(input_path),
            "hash": doc_hash,
            "timestamp": datetime.now().isoformat(),
            "total_chunks": total_blocks,
            "refine_guardrails": getattr(cfg, "refine_guardrails", "strict"),
        }
        state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    progress = _prepare_progress(
        progress_path=progress_path,
        resume_manifest=resume_manifest,
        total_blocks=total_blocks,
        logger=logger,
    )
    _write_progress(progress, logger)

    if debug_chunks:
        debug_file_path = output_path.with_name(f"{output_path.stem}_chunks_debug.jsonl")
        debug_file = debug_file_path.open("w", encoding="utf-8")

    def _write_chunk_debug(entry: dict) -> None:
        if debug_file:
            debug_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    refined_sections: List[str] = []
    with processing_context(stats, progress):
        for idx, (title, body) in enumerate(sections, start=1):
            refined_sections.append(
                refine_section(
                    title=title,
                    body=body,
                    backend=backend,
                    cfg=cfg,
                    logger=logger,
                    index=idx,
                    total=len(sections),
                    glossary_state=glossary_state,
                glossary_prompt_limit=glossary_prompt_limit,
                debug_refine=debug_refine,
                metrics=metrics,
                seen_chunks=seen_chunks,
                debug_writer=_write_chunk_debug if debug_chunks else None,
            )
        )

    final_md = "\n\n".join(refined_sections).strip()
    if not final_md:
        raise ValueError(f"Refine produziu texto vazio para {input_path}")

    final_md = sanitize_refine_output(final_md)

    write_text(output_path, final_md)
    if glossary_state:
        save_dynamic_glossary(glossary_state, logger)
    try:
        version = (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        version = "unknown"
    report = {
        "mode": "refine",
        "input": str(input_path),
        "total_chunks": stats.total_blocks,
        "cache_hits": metrics.get("cache_hits", 0),
        "fallbacks": metrics.get("fallbacks", 0),
        "collapse_detected": metrics.get("collapse", 0),
        "duplicates_reused": metrics.get("duplicates", 0),
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": version,
        "refine_guardrails": getattr(cfg, "refine_guardrails", "strict"),
        "effective_refine_chunk_chars": cfg.refine_chunk_chars,
        "max_chunk_chars_observed": metrics.get("max_chunk_chars_observed", 0),
    }
    try:
        (output_path.parent / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        refine_metrics = {
            "total_blocks": stats.total_blocks,
            "cache_hits": metrics.get("cache_hits", 0),
            "duplicates": metrics.get("duplicates", 0),
            "fallbacks": metrics.get("fallbacks", 0),
            "collapse": metrics.get("collapse", 0),
            "blocks": metrics.get("block_metrics", []),
            "guardrails_mode": getattr(cfg, "refine_guardrails", "strict"),
            "cleanup_mode": metrics.get("cleanup_mode", "off"),
            "cleanup_applied": metrics.get("cleanup_applied", False),
            "cleanup_stats": metrics.get("cleanup_stats", {}),
            "cleanup_preview_hash_before": metrics.get("cleanup_preview_hash_before"),
            "cleanup_preview_hash_after": metrics.get("cleanup_preview_hash_after"),
            "effective_refine_chunk_chars": cfg.refine_chunk_chars,
            "max_chunk_chars_observed": metrics.get("max_chunk_chars_observed", 0),
        }
        slug = Path(input_path).stem
        metrics_path = output_path.parent / f"{slug}_refine_metrics.json"
        metrics_path.write_text(json.dumps(refine_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    if debug_file:
        debug_file.close()
        if debug_file_path:
            logger.info("Arquivo de debug de refine: %s", debug_file_path)
    if debug_file:
        debug_file.close()
        if debug_file_path:
            logger.info("Arquivo de debug de refine: %s", debug_file_path)
    logger.info(
        "Refine concluído: %s (blocos: total=%d sucesso=%d placeholders=%d)",
        output_path.name,
        stats.total_blocks,
        stats.success_blocks,
        stats.error_blocks,
    )


def _call_with_retry(
    backend: LLMBackend,
    prompt: str,
    cfg: AppConfig,
    logger: logging.Logger,
    label: str,
    max_retries: int | None = None,
) -> tuple[str, str]:
    delay = cfg.initial_backoff
    last_error: Exception | None = None
    attempts = max_retries if max_retries is not None else cfg.max_retries
    for attempt in range(1, attempts + 1):
        try:
            latency, response = timed(backend.generate, prompt)
            raw_text = response.text
            text = sanitize_refine_output(raw_text)
            if not text.strip():
                raise ValueError("Texto vazio após sanitização do refine.")
            logger.info("%s ok (%.2fs, %d chars)", label, latency, len(text))
            return raw_text, text
        except Exception as exc:
            last_error = exc
            logger.warning("%s falhou (tentativa %d/%d): %s", label, attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(delay)
                delay *= cfg.backoff_factor
    raise RuntimeError(f"{label} falhou após {attempts} tentativas: {last_error}")
