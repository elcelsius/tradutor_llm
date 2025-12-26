"""
Pipeline de traducao em lotes.
"""

from __future__ import annotations

import json
import logging
import re
import time
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from .config import AppConfig
from .cache_utils import (
    cache_exists,
    chunk_hash,
    detect_model_collapse,
    load_cache,
    save_cache,
    is_near_duplicate,
    set_cache_base_dir,
)
from .llm_backend import LLMBackend
from .preprocess import (
    chunk_for_translation,
    paragraphs_from_text,
    preprocess_text,
)
from .section_splitter import split_into_sections
from .glossary_utils import format_manual_pairs_for_translation, select_terms_for_chunk
from .sanitizer import log_report, sanitize_translation_output, SanitizationReport
from .utils import timed
from .refine import has_suspicious_repetition  # reuse guardrail
from .anti_hallucination import anti_hallucination_filter
from .qa import needs_retry, count_quotes, count_quote_lines
from .postprocess_translation import postprocess_translation
from .quote_fix import fix_unbalanced_quotes, count_curly_quotes


def _extract_last_sentence(text: str) -> str:
    """Extrai a ultima frase simples (delimitada por .!?) e limpa marcadores."""
    cleaned = re.sub(r"###\s*TEXTO_TRADUZIDO_[A-Z_]*", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    for part in reversed(parts):
        candidate = part.strip().strip("#").strip()
        if candidate:
            return candidate
    return ""


def build_translation_prompt(
    chunk: str,
    context: str | None = None,
    glossary_text: str | None = None,
    allow_adaptation: bool = False,
) -> str:
    """Prompt minimalista para traducao EN -> PT-BR com delimitadores, contexto e glossario manual opcional."""
    context_block = ""
    if context:
        context_block = (
            "CONTEXT (DO NOT TRANSLATE OR REWRITE):\n"
            f"\"{context.strip()}\"\n\n"
        )

    glossary_block = ""
    if glossary_text:
        glossary_block = (
            "VOCE DEVE SEGUIR EXATAMENTE AS TRADUCOES OFICIAIS DO GLOSSARIO ABAIXO.\n"
            "NAO DEVE CRIAR OUTRAS VERSOES. NAO DEVE ALTERAR NOMES PROPRIOS.\n"
            "NAO DEVE ADICIONAR EXPLICACOES.\n"
            f"{glossary_text}\n\n"
        )

    adaptation_block = ""
    if allow_adaptation:
        adaptation_block = """
6. Em piadas e trocadilhos, manter o espírito e não a literalidade:

   * "Four Holey Smell-ders" → "Os Quatro Furados Fedorentos"
   * "Captain Mammaries" → "Capitã Peituda"
"""

    return f"""
Você é um TRADUTOR PROFISSIONAL DE LIGHT NOVELS, especializado em inglês → português brasileiro.
Sua tarefa é traduzir fielmente, com naturalidade e fluidez, sem alterar absolutamente nenhum evento, ordem narrativa, personalidade dos personagens ou conteúdo do original.

REGRAS PRINCIPAIS:

1. Tradução 100% fiel ao sentido.
2. NÃO resumir.
3. NÃO pular frases.
4. NÃO adicionar conteúdo.
5. NÃO mudar tom ou personalidade dos personagens.
6. Manter nomes exatos conforme glossário.
7. Manter número/pessoa corretos: não inverter singular/plural; não use "vocês" quando o original está no singular.

MELHORIAS OBRIGATÓRIAS:

1. Naturalizar português brasileiro sem calques.
2. Corrigir gagueiras e hesitações:

   * "H-he..." → "E-ele..." ou "Ele..."
3. Converter possessivos ingleses:

   * "Kirihara's group" → "grupo do Kirihara"
4. Remover ruído de OCR e PDF:

   * aspas triplas \"\"\" → remover
   * caracteres soltos → remover
5. Adaptar calques literais:

   * "What’s with that high-and-mighty attitude?!" → "Que atitude é essa, todo se achando?!"
{adaptation_block}7. Garantir fluidez e tom de light novel brasileira.

FORMATO DE SAÍDA:
Retorne exclusivamente:

### TEXTO_TRADUZIDO_INICIO

<texto traduzido>
### TEXTO_TRADUZIDO_FIM

Nada antes ou depois dos marcadores.

{glossary_block}{context_block}TEXTO A SER TRADUZIDO:
\"\"\"{chunk}\"\"\""""


def _parse_translation_output(raw: str) -> str:
    """Extrai bloco entre TEXTO_TRADUZIDO_INICIO/FIM; fallback para texto inteiro."""
    start = raw.find("### TEXTO_TRADUZIDO_INICIO")
    end = raw.find("### TEXTO_TRADUZIDO_FIM")
    if start != -1 and end != -1 and end > start:
        content = raw[start + len("### TEXTO_TRADUZIDO_INICIO") : end]
        return content.strip()
    raise ValueError("Marcadores TEXTO_TRADUZIDO_INICIO/FIM ausentes na resposta do modelo.")


def _strip_translate_markers(text: str) -> str:
    """Remove qualquer linha/bloco com marcadores TEXTO_TRADUZIDO_ remanescentes."""
    lines = []
    for ln in text.splitlines():
        if "TEXTO_TRADUZIDO_" in ln:
            continue
        lines.append(ln)
    cleaned = "\n".join(lines)
    cleaned = re.sub(
        r"### TEXTO_TRADUZIDO_INICIO.*?(### TEXTO_TRADUZIDO_FIM)?",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def _split_dialogue_blocks(chunk: str) -> list[str]:
    """
    Divide o chunk em blocos menores priorizando linhas de diálogo (aspas/travessão).
    """
    blocks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                blocks.append(joined)
            buffer = []

    for raw_line in chunk.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith(('"', "“", "”", "’", "-", "—")):
            flush()
            blocks.append(line)
        else:
            buffer.append(line)
    flush()
    return [b for b in blocks if b.strip()]


def _build_chunk_glossary(
    manual_terms: list[dict] | None,
    chunk: str,
    *,
    match_limit: int,
    fallback_limit: int,
    logger: logging.Logger,
    chunk_index: int,
) -> tuple[str | None, int, int]:
    """
    Seleciona termos do glossário que aparecem no chunk.
    Retorna (glossary_text, matched_count, total_injetados).
    """
    if not manual_terms:
        return None, 0, 0
    selected, matched = select_terms_for_chunk(
        manual_terms,
        chunk,
        match_limit=match_limit,
        fallback_limit=fallback_limit,
    )
    glossary_text = format_manual_pairs_for_translation(selected, limit=len(selected) or None)
    injected = len(selected)
    if injected:
        logger.debug(
            "Glossario (chunk %d): injetados=%d matched=%d",
            chunk_index,
            injected,
            matched,
        )
    return glossary_text or None, matched, injected


def translate_document(
    pdf_text: str,
    backend: LLMBackend,
    cfg: AppConfig,
    logger: logging.Logger,
    source_slug: str | None = None,
    progress_path: Path | None = None,
    resume_manifest: dict | None = None,
    glossary_text: str | None = None,
    glossary_manual_terms: list[dict] | None = None,
    debug_translation: bool = False,
    parallel_workers: int = 1,
    debug_chunks: bool = False,
    already_preprocessed: bool = False,
    split_by_sections: bool | None = None,
    allow_adaptation: bool | None = None,
    fail_on_chunk_error: bool | None = None,
) -> str:
    """
    Executa pre-processamento (opcional), chunking e traducao por lotes com sanitizacao.
    """
    set_cache_base_dir(cfg.output_dir)
    split_flag = cfg.split_by_sections if split_by_sections is None else split_by_sections
    allow_adapt_flag = cfg.translate_allow_adaptation if allow_adaptation is None else allow_adaptation
    fail_on_error = cfg.fail_on_chunk_error if fail_on_chunk_error is None else fail_on_chunk_error
    if not hasattr(backend, "temperature"):
        backend.temperature = cfg.translate_temperature
    clean = pdf_text if already_preprocessed else preprocess_text(pdf_text, logger, skip_front_matter=cfg.skip_front_matter)
    doc_hash = chunk_hash(clean)
    sections = split_into_sections(clean) if split_flag else [{"title": "Full Text", "body": clean}]
    chunk_records = []
    for sidx, sec in enumerate(sections, start=1):
        paragraphs = paragraphs_from_text(sec["body"])
        sec_chunks = chunk_for_translation(paragraphs, max_chars=cfg.translate_chunk_chars, logger=logger)
        for ch in sec_chunks:
            chunk_records.append({"section": sidx, "title": sec.get("title", ""), "text": ch})
    chunks = [c["text"] for c in chunk_records]
    max_chunk_len = max((len(c["text"]) for c in chunk_records), default=0)
    logger.info(
        "Iniciando traducao: %d chunks (alvo=%d, max_observado=%d)",
        len(chunks),
        cfg.translate_chunk_chars,
        max_chunk_len,
    )
    if parallel_workers > 1:
        logger.info("Context chaining ativo; paralelismo ajustado para 1 na tradução.")
        parallel_workers = 1
    state_path = Path(cfg.output_dir) / "state_traducao.json"
    dialogue_guardrails_mode = getattr(cfg, "translate_dialogue_guardrails", "strict")
    dialogue_split_fallback = getattr(cfg, "translate_dialogue_split_fallback", True)
    dialogue_retry_temps = getattr(cfg, "translate_dialogue_retry_temps", None) or [
        cfg.translate_temperature,
        0.25,
        0.10,
    ]
    glossary_match_limit = getattr(cfg, "translate_glossary_match_limit", 80)
    glossary_fallback_limit = getattr(cfg, "translate_glossary_fallback_limit", 30)
    try:
        state_payload = {
            "input_file": source_slug or "document",
            "hash": doc_hash,
            "timestamp": datetime.now().isoformat(),
            "total_chunks": len(chunks),
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    if cfg.dump_chunks and chunks:
        slug = source_slug or "document"
        debug_path = Path(cfg.output_dir) / f"{slug}_chunks_debug.md"
        total = len(chunks)
        parts = []
        for idx, chunk in enumerate(chunks, start=1):
            parts.append(f"=== CHUNK {idx}/{total} ===")
            parts.append(chunk)
            parts.append("")  # linha em branco entre chunks
        debug_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
        logger.info("Chunks salvos em %s", debug_path)

    translated_chunks: List[str] = []
    total_chunks = len(chunks)
    translated_ok: set[int] = set()
    failed_chunks: set[int] = set()
    chunk_outputs: dict[int, str] = {}
    processed_indices: set[int] = set()
    cache_hits = 0
    fallbacks = 0
    collapse_detected = 0
    duplicate_reuse = 0
    seen_chunks: list[tuple[str, str]] = []
    contamination_count = 0
    error_count = 0
    orig_chars_total = 0
    sanitized_chars_total = 0
    paragraph_mismatch: dict[str, int] | None = None
    chunk_metrics: list[dict] = []

    glossary_hash = chunk_hash(glossary_text) if glossary_text else None
    current_cache_signature = {
        "backend": getattr(backend, "backend", None),
        "model": getattr(backend, "model", None),
        "num_predict": getattr(backend, "num_predict", None),
        "temperature": getattr(backend, "temperature", None),
        "repeat_penalty": getattr(backend, "repeat_penalty", None),
        "translate_chunk_chars": cfg.translate_chunk_chars,
        "glossary_hash": glossary_hash,
    }

    def _is_cache_compatible(data: dict) -> bool:
        meta = data.get("metadata")
        if not isinstance(meta, dict):
            return False
        return all(meta.get(k) == v for k, v in current_cache_signature.items())

    if resume_manifest:
        manifest_total = resume_manifest.get("total_chunks")
        if isinstance(manifest_total, int) and manifest_total != total_chunks:
            logger.warning(
                "Manifesto indica %d chunks, mas chunking atual gerou %d; usando chunking atual.",
                manifest_total,
                total_chunks,
            )
        raw_chunks = resume_manifest.get("chunks") or {}
        if isinstance(raw_chunks, dict):
            for key, val in raw_chunks.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if isinstance(val, str):
                    chunk_outputs[idx] = val

        raw_translated = resume_manifest.get("translated_chunks") or []
        for idx in raw_translated:
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            if idx_int in chunk_outputs:
                translated_ok.add(idx_int)
            else:
                logger.warning(
                    "Manifesto marca chunk %s como traduzido, mas nao ha conteudo salvo; retraduzindo.",
                    idx_int,
                )

        raw_failed = resume_manifest.get("failed_chunks") or []
        for idx in raw_failed:
            try:
                failed_chunks.add(int(idx))
            except (TypeError, ValueError):
                continue

    def _write_progress() -> None:
        if progress_path is None:
            return
        data = {
            "total_chunks": total_chunks,
            "translated_chunks": sorted(translated_ok),
            "failed_chunks": sorted(failed_chunks),
            "timestamp": datetime.now().isoformat(),
            "chunks": {str(idx): text for idx, text in chunk_outputs.items()},
        }
        try:
            progress_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - I/O edge case
            logger.warning("Falha ao gravar manifesto de progresso em %s: %s", progress_path, exc)

    _write_progress()

    previous_context: str | None = None
    current_section: int | None = None
    debug_dir = Path(cfg.output_dir) / "debug_traducao"
    chunk_offsets: list[tuple[int | None, int | None]] = [(None, None)] * len(chunks)

    debug_file = None
    debug_file_path: Path | None = None
    if debug_chunks:
        debug_dir_path = Path(cfg.output_dir)
        debug_dir_path.mkdir(parents=True, exist_ok=True)
        base = source_slug or "document"
        debug_file_path = debug_dir_path / f"{base}_pt_chunks_debug.jsonl"
        debug_file = debug_file_path.open("w", encoding="utf-8")

    def _write_chunk_debug(entry: dict) -> None:
        if debug_file:
            debug_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _count_quotes(txt: str) -> int:
        return sum(1 for ch in txt if ch in {'"', "“", "”", "'"})

    
    for idx, chunk_info in enumerate(chunk_records, start=1):
        chunk = chunk_info["text"]
        chunk_glossary_text, glossary_matched, glossary_injected = _build_chunk_glossary(
            glossary_manual_terms,
            chunk,
            match_limit=glossary_match_limit,
            fallback_limit=glossary_fallback_limit,
            logger=logger,
            chunk_index=idx,
        )
        if not chunk_glossary_text:
            chunk_glossary_text = glossary_text
        section_id = chunk_info.get("section")
        if current_section is None or section_id != current_section:
            previous_context = None
            current_section = section_id
        h = chunk_hash(chunk)
        start_offset, end_offset = chunk_offsets[idx - 1] if idx - 1 < len(chunk_offsets) else (None, None)
        from_cache = False
        from_duplicate = False
        llm_attempts = 0
        raw_text: str | None = None
        sanitizer_report = None
        error_message: str | None = None
        parsed_clean: str | None = None

        if cache_exists("translate", h):
            data = load_cache("translate", h)
            meta_ok = _is_cache_compatible(data)
            if not meta_ok:
                logger.debug("Cache de tradução ignorado: assinatura diferente de backend/model/num_predict.")
            else:
                cached = data.get("final_output")
                if cached:
                    logger.info("Reusando cache de tradução para chunk trad-%d/%d", idx, total_chunks)
                    parsed_clean = cached
                    translated_chunks.append(cached)
                    translated_ok.add(idx)
                    chunk_outputs[idx] = cached
                    processed_indices.add(idx)
                    cache_hits += 1
                    from_cache = True
                    previous_context = _extract_last_sentence(chunk)
                    _write_progress()

        if parsed_clean is None:
            if idx in translated_ok and idx in chunk_outputs:
                logger.info("Reusando traducao salva para chunk trad-%d/%d", idx, total_chunks)
                parsed_clean = chunk_outputs[idx]
                translated_chunks.append(chunk_outputs[idx])
                processed_indices.add(idx)
                previous_context = _extract_last_sentence(chunk)
                _write_progress()
            else:
                reused_dup = False
                for prev_chunk, prev_final in seen_chunks:
                    if is_near_duplicate(prev_chunk, chunk):
                        logger.info("Chunk %d marcado como duplicado de um anterior; reuso habilitado.", idx)
                        parsed_clean = prev_final
                        translated_chunks.append(prev_final)
                        translated_ok.add(idx)
                        chunk_outputs[idx] = prev_final
                        processed_indices.add(idx)
                        duplicate_reuse += 1
                        from_duplicate = True
                        previous_context = _extract_last_sentence(chunk)
                        _write_progress()
                        reused_dup = True
                        break
                if not reused_dup:
                    base_prompt = build_translation_prompt(
                        chunk,
                        context=previous_context,
                        glossary_text=chunk_glossary_text,
                        allow_adaptation=allow_adapt_flag,
                    )
                    prompt = base_prompt
                    try:
                        attempt = 0
                        retry_reason = ""
                        while attempt < cfg.max_retries:
                            prev_temp = backend.temperature
                            temp_for_attempt = dialogue_retry_temps[min(attempt, len(dialogue_retry_temps) - 1)]
                            backend.temperature = temp_for_attempt
                            try:
                                raw_text, _clean_text, llm_attempts, sanitizer_report = _call_with_retry(
                                    backend=backend,
                                    prompt=prompt,
                                    cfg=cfg,
                                    logger=logger,
                                    label=f"trad-{idx}/{len(chunks)}",
                                )
                            finally:
                                backend.temperature = prev_temp
                            parsed = _parse_translation_output(raw_text)
                            parsed_raw = _strip_translate_markers(parsed)
                            parsed_clean, report = sanitize_translation_output(parsed_raw, logger=logger, fail_on_contamination=False)
                            sanitizer_report = report
                            log_report(report, logger, prefix=f"trad-parse-{idx}")
                            if debug_translation and report.contamination_detected:
                                debug_dir.mkdir(parents=True, exist_ok=True)
                                attempt_tag = attempt + 1
                                base = f"chunk{idx:03d}_attempt{attempt_tag}"
                                (debug_dir / f"{base}_raw.txt").write_text(parsed_raw, encoding="utf-8")
                                (debug_dir / f"{base}_clean.txt").write_text(parsed_clean, encoding="utf-8")
                            if not parsed_clean.strip():
                                raise ValueError("Traducao vazia apos parsing/sanitizacao.")
                            raw_candidate = anti_hallucination_filter(orig=chunk, llm_raw=raw_text, cleaned=parsed_raw, mode="translate")
                            parsed_clean = anti_hallucination_filter(orig=chunk, llm_raw=raw_text, cleaned=parsed_clean, mode="translate")
                            sanitized_ratio = len(parsed_clean.strip()) / max(len(parsed_raw.strip()), 1) if parsed_raw.strip() else 1.0
                            iq = _count_quotes(chunk)
                            oq = _count_quotes(parsed_clean)
                            iql = count_quote_lines(chunk)
                            oql = count_quote_lines(parsed_clean)
                            clean_retry, clean_reason = needs_retry(
                                chunk,
                                parsed_clean,
                                input_quotes=iq,
                                output_quotes=oq,
                                input_quote_lines=iql,
                                output_quote_lines=oql,
                                contamination_detected=bool(report.contamination_detected),
                                sanitization_ratio=sanitized_ratio,
                            )
                            raw_retry, _ = needs_retry(chunk, raw_candidate, input_quotes=iq, output_quotes=_count_quotes(raw_candidate), input_quote_lines=iql, output_quote_lines=count_quote_lines(raw_candidate), contamination_detected=False, sanitization_ratio=1.0)
                            prefer_raw = report.contamination_detected and not raw_retry and (sanitized_ratio < 0.95 or "omissao_dialogo" in clean_reason)
                            retry = clean_retry or (report.contamination_detected and sanitized_ratio < 0.95) or (report.contamination_detected and "omissao_dialogo" in clean_reason)
                            retry_reason = clean_reason if clean_retry else ("sanitizacao_agressiva" if sanitized_ratio < 0.95 else retry_reason)
                            guardrail_triggered = False
                            guardrail_reason = ""
                            if dialogue_guardrails_mode != "off":
                                guard_ratio = 0.5 if dialogue_guardrails_mode == "relaxed" else 0.4
                                if iq >= 4 and oq < max(1, int(iq * guard_ratio)):
                                    guardrail_triggered = True
                                    guardrail_reason = f"omissao_dialogo_guardrail_quotes ({oq}/{iq})"
                                elif iql >= 2 and oql < max(1, int(iql * guard_ratio)):
                                    guardrail_triggered = True
                                    guardrail_reason = f"omissao_dialogo_guardrail_linhas ({oql}/{iql})"
                            if guardrail_triggered:
                                retry = True
                                retry_reason = guardrail_reason or retry_reason or "omissao_dialogo_guardrail"
                            if not retry:
                                # se sanitização falhou mas raw está ok, preferir raw
                                if prefer_raw:
                                    parsed_clean = raw_candidate
                                break
                            attempt += 1
                            is_dialogue_retry = "omissao_dialogo" in (retry_reason or "") or guardrail_triggered
                            if attempt >= cfg.max_retries and is_dialogue_retry and dialogue_split_fallback:
                                blocks = _split_dialogue_blocks(chunk) or [chunk]
                                logger.warning(
                                    "Fallback de split de dialogos no chunk %d/%d (%d blocos).",
                                    idx,
                                    len(chunks),
                                    len(blocks),
                                )
                                block_outputs: list[str] = []
                                for b_idx, block in enumerate(blocks, start=1):
                                    block_glossary, _, _ = _build_chunk_glossary(
                                        glossary_manual_terms,
                                        block,
                                        match_limit=glossary_match_limit,
                                        fallback_limit=glossary_fallback_limit,
                                        logger=logger,
                                        chunk_index=idx,
                                    )
                                    if not block_glossary:
                                        block_glossary = chunk_glossary_text
                                    block_prompt = build_translation_prompt(
                                        block,
                                        context=None,
                                        glossary_text=block_glossary,
                                        allow_adaptation=allow_adapt_flag,
                                    )
                                    block_prompt += "\n\nATENÇÃO: NENHUMA fala pode ser omitida. Traduza exatamente este bloco preservando todas as aspas e travessões. Não resuma."
                                    prev_temp_block = backend.temperature
                                    backend.temperature = dialogue_retry_temps[-1]
                                    try:
                                        block_raw, _block_clean, _, block_report = _call_with_retry(
                                            backend=backend,
                                            prompt=block_prompt,
                                            cfg=cfg,
                                            logger=logger,
                                            label=f"trad-split-{idx}-{b_idx}",
                                        )
                                    finally:
                                        backend.temperature = prev_temp_block
                                    block_parsed = _parse_translation_output(block_raw)
                                    block_parsed_raw = _strip_translate_markers(block_parsed)
                                    block_clean, block_clean_report = sanitize_translation_output(
                                        block_parsed_raw, logger=logger, fail_on_contamination=False
                                    )
                                    sanitizer_report = block_clean_report
                                    block_clean = anti_hallucination_filter(orig=block, llm_raw=block_raw, cleaned=block_clean, mode="translate")
                                    block_clean = postprocess_translation(block_clean, block)
                                    block_outputs.append(block_clean)
                                parsed_clean = "\n\n".join(block_outputs).strip()
                                retry = False
                                break
                            if attempt >= cfg.max_retries:
                                if prefer_raw:
                                    parsed_clean = raw_candidate
                                break
                            logger.warning(
                                "QA retry traducao chunk %d/%d: %s (tentativa %d/%d)",
                                idx,
                                len(chunks),
                                retry_reason,
                                attempt + 1,
                                cfg.max_retries,
                            )
                            if "omissao_dialogo" in retry_reason:
                                prompt = base_prompt + "\n\nATENÇÃO: Você omitiu falas. Refaça traduzindo TODAS as frases e mantendo cada fala entre aspas exatamente uma vez. Não resuma. Não remova risos/interjeições."
                            elif "truncado" in retry_reason:
                                prompt = base_prompt + "\n\nATENÇÃO: Sua saída foi truncada. Refaça incluindo TODO o conteúdo."
                            else:
                                prompt = base_prompt + "\n\nATENÇÃO: Sua saída anterior veio truncada ou repetitiva. Refaça e inclua TODO o conteúdo. Não resuma."

                        parsed_clean = postprocess_translation(parsed_clean, chunk)
                        # correção de aspas curvas
                        opens_q, closes_q = count_curly_quotes(parsed_clean)
                        if opens_q != closes_q:
                            parsed_clean, fixed = fix_unbalanced_quotes(parsed_clean, logger=logger, label=f"trad-{idx}")
                            if fixed:
                                opens_q, closes_q = count_curly_quotes(parsed_clean)
                        orig_len = len(chunk.strip())
                        cleaned_len = len(parsed_clean.strip())
                        if orig_len and cleaned_len < orig_len * 0.5:
                            logger.error(
                                "Traducao suspeita: chunk %d/%d ficou com %d%% do tamanho original apos sanitizacao.",
                                idx,
                                len(chunks),
                                int((cleaned_len / orig_len) * 100) if orig_len else 0,
                            )
                            marker = f"[CHUNK_TRADUCAO_SUSPEITO_{idx}] "
                            parsed_clean = f"{marker}{parsed_clean}" if parsed_clean.strip() else marker
                        elif orig_len and cleaned_len < orig_len * 0.7:
                            logger.warning(
                                "Traducao suspeita: chunk %d/%d muito menor que o original; mantendo traducao mesmo assim.",
                                idx,
                                len(chunks),
                            )
                        if has_suspicious_repetition(parsed_clean):
                            logger.warning(
                                "Traducao com repeticao suspeita; chunk %d/%d marcado para revisao.",
                                idx,
                                len(chunks),
                            )
                        if detect_model_collapse(parsed_clean, original_len=len(chunk), mode="translate"):
                            logger.warning(
                                "Colapso detectado no chunk %d/%d; usando texto original do chunk.",
                                idx,
                                len(chunks),
                            )
                            collapse_detected += 1
                        translated_chunks.append(parsed_clean)
                        translated_ok.add(idx)
                        failed_chunks.discard(idx)
                        chunk_outputs[idx] = parsed_clean
                        processed_indices.add(idx)
                        seen_chunks.append((chunk, parsed_clean))
                        save_cache(
                            "translate",
                            h,
                            raw_output=raw_text,
                            final_output=parsed_clean,
                            metadata={
                                "chunk_index": idx,
                                "mode": "translate",
                                "source": source_slug or "",
                                "backend": getattr(backend, "backend", None),
                                "model": getattr(backend, "model", None),
                                "num_predict": getattr(backend, "num_predict", None),
                                "temperature": getattr(backend, "temperature", None),
                                "repeat_penalty": getattr(backend, "repeat_penalty", None),
                                "translate_chunk_chars": cfg.translate_chunk_chars,
                                "glossary_hash": glossary_hash,
                            },
                        )
                        if debug_translation and idx <= 5:
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            base = f"chunk{idx:03d}"
                            (debug_dir / f"{base}_original_en.txt").write_text(chunk, encoding="utf-8")
                            (debug_dir / f"{base}_context.txt").write_text(previous_context or "", encoding="utf-8")
                            (debug_dir / f"{base}_llm_raw.txt").write_text(raw_text, encoding="utf-8")
                            (debug_dir / f"{base}_final_pt.txt").write_text(parsed_clean, encoding="utf-8")
                    except Exception as exc:
                        # debug de falha
                        fail_dir = Path(cfg.output_dir) / "debug_translate_chunks" / "failed"
                        try:
                            fail_dir.mkdir(parents=True, exist_ok=True)
                            attempt_tag = llm_attempts or 0
                            if raw_text:
                                (fail_dir / f"chunk{idx:03d}_attempt{attempt_tag}_raw.txt").write_text(raw_text, encoding="utf-8")
                            (fail_dir / f"chunk{idx:03d}_prompt.txt").write_text(prompt, encoding="utf-8")
                            (fail_dir / f"chunk{idx:03d}_context.txt").write_text(previous_context or "", encoding="utf-8")
                            error_payload = {
                                "chunk_index": idx,
                                "label": f"trad-{idx}/{len(chunks)}",
                                "error": str(exc),
                                "stack": traceback.format_exc(),
                            }
                            (fail_dir / f"chunk{idx:03d}_error.json").write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                        failed_chunks.add(idx)
                        error_message = str(exc)
                        llm_attempts = getattr(exc, "attempts", llm_attempts)
                        if sanitizer_report is None and hasattr(exc, "last_report"):
                            sanitizer_report = getattr(exc, "last_report")
                        raise RuntimeError(f"Falha ao traduzir chunk {idx}/{len(chunks)}: {exc}") from exc
                    finally:
                        previous_context = _extract_last_sentence(chunk)
                        _write_progress()

        final_output = parsed_clean if parsed_clean is not None else ""
        orig_len_for_stats = len(chunk)
        orig_chars_total += orig_len_for_stats
        sanitized_chars_total += len(final_output)
        if sanitizer_report and sanitizer_report.contamination_detected:
            contamination_count += 1
        if error_message:
            error_count += 1

        cleaned_ratio = (len(final_output.strip()) / max(len(chunk.strip()), 1)) if chunk.strip() else 0.0
        too_short = cleaned_ratio < 0.60
        too_long = cleaned_ratio > 1.80
        suspicious = has_suspicious_repetition(final_output)
        orig_quotes = _count_quotes(chunk)
        translated_quotes = _count_quotes(final_output)
        possible_omission = False
        if orig_quotes >= 4 and translated_quotes <= max(1, int(orig_quotes * 0.4)):
            possible_omission = True
            logger.warning(
                "Possível omissão de falas no chunk %d/%d (aspas %d -> %d).",
                idx,
                total_chunks,
                orig_quotes,
                translated_quotes,
            )
        chunk_metrics.append(
            {
                "chunk_index": idx,
                "chars_in": len(chunk),
                "chars_out": len(final_output),
                "ratio_out_in": round(cleaned_ratio, 3),
                "from_cache": from_cache,
                "from_duplicate": from_duplicate,
                "llm_attempts": llm_attempts,
                "too_short": too_short,
                "too_long": too_long,
                "suspicious_repetition": suspicious,
                "possible_omission": possible_omission,
            }
        )

        report_dict = {
            "contamination_detected": bool(sanitizer_report.contamination_detected) if sanitizer_report else False,
            "removed_lines_count": getattr(sanitizer_report, "removed_lines_count", 0) if sanitizer_report else 0,
            "collapsed_repetitions": getattr(sanitizer_report, "collapsed_repetitions", 0) if sanitizer_report else 0,
            "leading_noise_removed": getattr(sanitizer_report, "leading_noise_removed", False) if sanitizer_report else False,
            "removed_think_blocks": getattr(sanitizer_report, "removed_think_blocks", 0) if sanitizer_report else 0,
        }

        if debug_chunks:
            entry = {
                "chunk_index": idx,
                "original_start_offset": start_offset,
                "original_end_offset": end_offset,
                "original_text": chunk,
                "original_chars": orig_len_for_stats,
                "original_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                "from_cache": from_cache,
                "from_duplicate": from_duplicate,
                "llm_attempts": llm_attempts,
                "llm_raw_output": raw_text,
                "sanitized_output": final_output,
                "sanitized_chars": len(final_output),
                "sanitized_hash": hashlib.sha256(final_output.encode("utf-8")).hexdigest(),
                "sanitizer_report": report_dict,
                "error": error_message,
            }
            _write_chunk_debug(entry)
    logger.info(
        "Resumo da traducao: total=%d sucesso=%d erro=%d",
        total_chunks,
        len(translated_ok),
        len(failed_chunks),
    )

    if len(processed_indices) != total_chunks:
        logger.error("Inconsistencia: apenas %d/%d chunks registraram alguma saida.", len(processed_indices), total_chunks)

    missing_outputs = [i for i in range(1, total_chunks + 1) if i not in chunk_outputs]
    if missing_outputs:
        logger.error("Chunks sem saida detectados apos traducao: %s; placeholders inseridos.", missing_outputs)
        for midx in missing_outputs:
            placeholder = f"[CHUNK_NAO_PROCESSADO_{midx}]"
            chunk_outputs[midx] = placeholder
            failed_chunks.add(midx)
        _write_progress()

    ordered_outputs = [chunk_outputs.get(i, f"[CHUNK_NAO_PROCESSADO_{i}]") for i in range(1, total_chunks + 1)]
    translated_chunks = ordered_outputs

    result = "\n\n".join(ordered_outputs).strip()
    if not result:
        raise ValueError("Traducao resultou em texto vazio.")

    translated_paragraphs = [p for p in result.split("\n\n") if p.strip()]
    if len(translated_paragraphs) < len(paragraphs):
        logger.error(
            "Paragrafos ausentes apos traducao: original=%d traduzido=%d",
            len(paragraphs),
            len(translated_paragraphs),
        )
        paragraph_mismatch = {"original": len(paragraphs), "translated": len(translated_paragraphs)}

    if debug_chunks:
        reduction_pct = (sanitized_chars_total / orig_chars_total * 100) if orig_chars_total else 0.0
        avg_orig = (orig_chars_total / total_chunks) if total_chunks else 0.0
        avg_san = (sanitized_chars_total / total_chunks) if total_chunks else 0.0
        logger.info(
            "Debug chunks resumo: total=%d cache=%d dup=%d contaminados=%d erros=%d",
            total_chunks,
            cache_hits,
            duplicate_reuse,
            contamination_count,
            error_count,
        )
        logger.info(
            "Debug chunks tamanhos: orig_med=%.1f san_med=%.1f (%.1f%% do original)",
            avg_orig,
            avg_san,
            reduction_pct,
        )
        if debug_file_path:
            logger.info("Arquivo de debug de chunks: %s", debug_file_path)

    result, final_report = sanitize_translation_output(result, logger=logger, fail_on_contamination=False)
    opens_final, closes_final = count_curly_quotes(result)
    if opens_final != closes_final:
        result, _ = fix_unbalanced_quotes(result, logger=logger, label="trad-final")
    log_report(final_report, logger, prefix="trad-final")
    try:
        version = (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        version = "unknown"
    status = "failed" if failed_chunks else "ok"
    report = {
        "mode": "translate",
        "status": status,
        "input": source_slug or "",
        "total_chunks": total_chunks,
        "cache_hits": cache_hits,
        "fallbacks": fallbacks,
        "failed_chunks": len(failed_chunks),
        "collapse_detected": collapse_detected,
        "duplicates_reused": duplicate_reuse,
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": version,
        "effective_translate_chunk_chars": cfg.translate_chunk_chars,
        "max_chunk_chars_observed": max_chunk_len,
    }
    if paragraph_mismatch:
        report["paragraph_mismatch"] = paragraph_mismatch
    try:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        slug = (source_slug or "document").replace("\\", "_").replace("/", "_")
        report_path = Path(cfg.output_dir) / f"{slug}_translate_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_payload = {
            "total_chunks": total_chunks,
            "cache_hits": cache_hits,
            "duplicates_reused": duplicate_reuse,
            "fallbacks": fallbacks,
            "failed_chunks": len(failed_chunks),
            "collapse_detected": collapse_detected,
            "chunks": chunk_metrics,
            "effective_translate_chunk_chars": cfg.translate_chunk_chars,
            "max_chunk_chars_observed": max_chunk_len,
        }
        metrics_path = Path(cfg.output_dir) / f"{slug}_translate_metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    if debug_file:
        debug_file.close()
        if debug_file_path:
            logger.info("Arquivo de debug de chunks: %s", debug_file_path)
    if failed_chunks:
        msg = (
            f"Traducao finalizada com falhas: {len(failed_chunks)}/{total_chunks} chunks nao foram traduzidos. "
            "Placeholders foram inseridos; consulte debug_translate_chunks/failed."
        )
        if fail_on_error:
            raise RuntimeError(
                msg
            )
        logger.error(msg)
    return result


def _call_with_retry(
    backend: LLMBackend,
    prompt: str,
    cfg: AppConfig,
    logger: logging.Logger,
    label: str,
) -> tuple[str, str, int, SanitizationReport | None]:
    """Chama backend com retry e sanitizacao leve para traducao."""
    delay = cfg.initial_backoff
    last_error: Exception | None = None
    last_report: SanitizationReport | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            latency, response = timed(backend.generate, prompt)
            text, report = sanitize_translation_output(response.text, logger=logger, fail_on_contamination=False)
            last_report = report
            log_report(report, logger, prefix=label)
            if not text.strip():
                raise ValueError("Texto vazio apos sanitizacao.")
            if report.contamination_detected:
                logger.warning(
                    "%s: contaminacao detectada; texto limpo sera usado (%d chars)",
                    label,
                    len(text),
                )
            logger.info("%s ok (%.2fs, %d chars)", label, latency, len(text))
            return response.text, text, attempt, report
        except Exception as exc:
            last_error = exc
            logger.warning("%s falhou (tentativa %d/%d): %s", label, attempt, cfg.max_retries, exc)
            if attempt < cfg.max_retries:
                time.sleep(delay)
                delay *= cfg.backoff_factor
    err = RuntimeError(f"{label} falhou apos {cfg.max_retries} tentativas: {last_error}")
    setattr(err, "attempts", cfg.max_retries)
    setattr(err, "last_report", last_report)
    raise err
