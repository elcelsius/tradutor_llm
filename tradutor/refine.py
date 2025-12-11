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
from typing import Dict, List, Tuple

from desquebrar import normalize_md_paragraphs

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
    return f"""
Você atuará como um POLIDOR MINIMALISTA.

Reescreva o texto abaixo sem alterar fatos, ordem, diálogos ou conteúdo narrativo.
Corrija apenas pequenos erros de digitação e vírgulas, e resolva artefatos de OCR/PDF.
NÃO resuma. NÃO expanda. NÃO interprete. NÃO remova ideias. NÃO adicione nada.
NÃO envolva a saída em molduras ou comentários. NÃO inclua glossários.
NÃO use "..." para representar conteúdo omitido. NÃO mude o idioma.
Preserve todos os parágrafos, falas e informações.

Formate sua resposta EXATAMENTE assim e nada mais:
### TEXTO_REFINADO_INICIO
<texto refinado>
### TEXTO_REFINADO_FIM

Texto para revisão (PT-BR):
\"\"\"{section}\"\"\"
"""


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
        h = chunk_hash(chunk)
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
                continue
        if cache_exists("refine", h):
            data = load_cache("refine", h)
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
                continue
        if progress and block_idx in progress.refined_blocks and block_idx in progress.chunk_outputs:
            logger.info("Reusando refinamento salvo para bloco ref-%d/%d-%d/%d", index, total, c_idx, len(chunks))
            refined_parts.append(progress.chunk_outputs[block_idx])
            if stats:
                stats.success_blocks += 1
            _write_progress(progress, logger)
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
            refined_text = response_text
            if glossary_state:
                refined_text, suggestion_block = split_refined_and_suggestions(llm_raw)
                suggestions = parse_glossary_suggestions(suggestion_block or "")
                if suggestions:
                    updated = apply_suggestions_to_state(glossary_state, suggestions, logger)
                    if updated:
                        save_dynamic_glossary(glossary_state, logger)
                        glossary_block = format_glossary_for_prompt(
                            glossary_state.combined_index,
                            glossary_prompt_limit,
                        )

            refined_text = anti_hallucination_filter(orig=chunk, llm_raw=llm_raw, cleaned=refined_text, mode="refine")
            if len(refined_text.strip()) < len(chunk.strip()) * 0.80:
                logger.warning(
                    "Refinador devolveu texto menor do que deveria; mantendo texto original (chunk %d/%d da seção %s).",
                    c_idx,
                    len(chunks),
                    title or f"#{index}",
                )
                refined_text = chunk
                metrics["fallbacks"] = metrics.get("fallbacks", 0) + 1
            if len(refined_text.strip()) > len(chunk.strip()) * 2.0:
                logger.warning(
                    "Refinador devolveu texto muito maior do que o original; mantendo texto original (chunk %d/%d da seção %s).",
                    c_idx,
                    len(chunks),
                    title or f"#{index}",
                )
                refined_text = chunk
                metrics["fallbacks"] = metrics.get("fallbacks", 0) + 1
            if has_suspicious_repetition(refined_text):
                logger.warning(
                    "Refinador devolveu texto com repetição suspeita; mantendo texto original (chunk %d/%d da seção %s).",
                    c_idx,
                    len(chunks),
                    title or f"#{index}",
                )
                refined_text = chunk
                metrics["fallbacks"] = metrics.get("fallbacks", 0) + 1
            if detect_model_collapse(refined_text, original_len=len(chunk), mode="refine"):
                logger.warning(
                    "Colapso detectado no chunk %d/%d da seção %s; mantendo texto original.",
                    c_idx,
                    len(chunks),
                    title or f"#{index}",
                )
                refined_text = chunk
                metrics["collapse"] = metrics.get("collapse", 0) + 1
                metrics["fallbacks"] = metrics.get("fallbacks", 0) + 1
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
            save_cache(
                "refine",
                h,
                raw_output=llm_raw,
                final_output=refined_text,
                metadata={"chunk_index": c_idx, "section_index": index, "mode": "refine"},
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
) -> None:
    md_text = read_text(input_path)
    if preprocess_advanced:
        md_text = advanced_clean(md_text)
    doc_hash = chunk_hash(md_text)
    if normalize_paragraphs:
        md_text = normalize_md_paragraphs(md_text)
    sections = split_markdown_sections(md_text)
    logger.info("Arquivo %s: %d seções detectadas", input_path.name, len(sections))
    stats = RefineStats()
    metrics: dict[str, int] = {"cache_hits": 0, "fallbacks": 0, "collapse": 0, "duplicates": 0}
    seen_chunks: list[tuple[str, str]] = []

    # Pré-computa total de blocos para progress
    total_blocks = 0
    for _, body in sections:
        paragraphs = paragraphs_from_text(body)
        chunks = chunk_for_refine(paragraphs, max_chars=cfg.refine_chunk_chars, logger=logger)
        total_blocks += len(chunks)

    if progress_path is None:
        progress_path = output_path.with_name(f"{output_path.stem}_progress.json")

    state_path = output_path.parent / "state_refine.json"
    try:
        state_payload = {
            "input_file": str(input_path),
            "hash": doc_hash,
            "timestamp": datetime.now().isoformat(),
            "total_chunks": total_blocks,
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
    }
    try:
        (output_path.parent / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
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
