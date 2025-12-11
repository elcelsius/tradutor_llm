"""
Pipeline de traducao em lotes.
"""

from __future__ import annotations

import json
import logging
import re
import time
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
)
from .llm_backend import LLMBackend
from .preprocess import (
    chunk_for_translation,
    paragraphs_from_text,
    preprocess_text,
)
from .glossary_utils import format_manual_pairs_for_translation
from .sanitizer import sanitize_text, log_report
from .utils import timed
from .refine import has_suspicious_repetition  # reuse guardrail
from .anti_hallucination import anti_hallucination_filter


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


def build_translation_prompt(chunk: str, context: str | None = None, glossary_text: str | None = None) -> str:
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

    return f"""
You are a professional translator. Translate the text from ENGLISH to BRAZILIAN PORTUGUESE.

Do not summarize. Do not add explanations, comments, or glossaries.
Do not invent new sentences or events. Do not skip any part of the original text.
Do not replace content with "...". Preserve paragraph breaks as much as possible.
Keep names and proper nouns as is unless a clear translation is standard.

Nao resuma o texto. Nao acrescente comentarios ou glossario.
Nao omita frases. Nao use "..." para pular partes do conteudo.
Preserve a ordem e o conteudo de todas as frases.

Your response must be EXACTLY in this format and nothing else:
### TEXTO_TRADUZIDO_INICIO
<traducao para PT-BR>
### TEXTO_TRADUZIDO_FIM

{glossary_block}{context_block}TEXTO A SER TRADUZIDO:
\"\"\"{chunk}\"\"\""""


def _parse_translation_output(raw: str) -> str:
    """Extrai bloco entre TEXTO_TRADUZIDO_INICIO/FIM; fallback para texto inteiro."""
    start = raw.find("### TEXTO_TRADUZIDO_INICIO")
    end = raw.find("### TEXTO_TRADUZIDO_FIM")
    if start != -1 and end != -1 and end > start:
        content = raw[start + len("### TEXTO_TRADUZIDO_INICIO") : end]
        return content.strip()
    return raw


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


def translate_document(
    pdf_text: str,
    backend: LLMBackend,
    cfg: AppConfig,
    logger: logging.Logger,
    source_slug: str | None = None,
    progress_path: Path | None = None,
    resume_manifest: dict | None = None,
    glossary_text: str | None = None,
    debug_translation: bool = False,
    parallel_workers: int = 1,
) -> str:
    """
    Executa pre-processamento, chunking e traducao por lotes com sanitizacao.
    """
    clean = preprocess_text(pdf_text, logger)
    doc_hash = chunk_hash(clean)
    paragraphs = paragraphs_from_text(clean)
    chunks = chunk_for_translation(paragraphs, max_chars=cfg.translate_chunk_chars, logger=logger)
    logger.info("Iniciando traducao: %d chunks", len(chunks))
    if parallel_workers > 1:
        logger.info("Context chaining ativo; paralelismo ajustado para 1 na tradução.")
        parallel_workers = 1
    state_path = Path(cfg.output_dir) / "state_traducao.json"
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
    cache_hits = 0
    fallbacks = 0
    collapse_detected = 0
    duplicate_reuse = 0
    seen_chunks: list[tuple[str, str]] = []

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
    debug_dir = Path(cfg.output_dir) / "debug_traducao"

    for idx, chunk in enumerate(chunks, start=1):
        h = chunk_hash(chunk)
        if cache_exists("translate", h):
            data = load_cache("translate", h)
            cached = data.get("final_output")
            if cached:
                logger.info("Reusando cache de tradução para chunk trad-%d/%d", idx, total_chunks)
                translated_chunks.append(cached)
                translated_ok.add(idx)
                chunk_outputs[idx] = cached
                cache_hits += 1
                previous_context = _extract_last_sentence(chunk)
                _write_progress()
                continue

        if idx in translated_ok and idx in chunk_outputs:
            logger.info("Reusando traducao salva para chunk trad-%d/%d", idx, total_chunks)
            translated_chunks.append(chunk_outputs[idx])
            previous_context = _extract_last_sentence(chunk)
            _write_progress()
            continue

        reused_dup = False
        for prev_chunk, prev_final in seen_chunks:
            if is_near_duplicate(prev_chunk, chunk):
                logger.info("Chunk %d marcado como duplicado de um anterior; reuso habilitado.", idx)
                translated_chunks.append(prev_final)
                translated_ok.add(idx)
                chunk_outputs[idx] = prev_final
                duplicate_reuse += 1
                previous_context = _extract_last_sentence(chunk)
                _write_progress()
                reused_dup = True
                break
        if reused_dup:
            continue

        prompt = build_translation_prompt(chunk, context=previous_context, glossary_text=glossary_text)
        try:
            raw_text, _clean_text = _call_with_retry(
                backend=backend,
                prompt=prompt,
                cfg=cfg,
                logger=logger,
                label=f"trad-{idx}/{len(chunks)}",
            )
            parsed = _parse_translation_output(raw_text)
            parsed = _strip_translate_markers(parsed)
            parsed_clean, report = sanitize_text(parsed, logger=logger, fail_on_contamination=False)
            log_report(report, logger, prefix=f"trad-parse-{idx}")
            if not parsed_clean.strip():
                raise ValueError("Traducao vazia apos parsing/sanitizacao.")
            parsed_clean = anti_hallucination_filter(orig=chunk, llm_raw=raw_text, cleaned=parsed_clean, mode="translate")
            if len(parsed_clean) < len(chunk) * 0.7:
                logger.warning(
                    "Traducao suspeita: chunk %d/%d muito menor que o original; mantendo traducao mesmo assim.",
                    idx,
                    len(chunks),
                )
                # não faz fallback para o original; apenas registra
            if has_suspicious_repetition(parsed_clean):
                logger.warning(
                    "Traducao com repeticao suspeita; chunk %d/%d marcado para revisao.",
                    idx,
                    len(chunks),
                )
                # não faz fallback; apenas registra
            if detect_model_collapse(parsed_clean, original_len=len(chunk), mode="translate"):
                logger.warning(
                    "Colapso detectado no chunk %d/%d; usando texto original do chunk.",
                    idx,
                    len(chunks),
                )
                # para tradução, não reverte automaticamente; apenas sinaliza
                collapse_detected += 1
            translated_chunks.append(parsed_clean)
            translated_ok.add(idx)
            failed_chunks.discard(idx)
            chunk_outputs[idx] = parsed_clean
            seen_chunks.append((chunk, parsed_clean))
            save_cache(
                "translate",
                h,
                raw_output=raw_text,
                final_output=parsed_clean,
                metadata={"chunk_index": idx, "mode": "translate", "source": source_slug or ""},
            )
            if debug_translation and idx <= 5:
                debug_dir.mkdir(parents=True, exist_ok=True)
                base = f"chunk{idx:03d}"
                (debug_dir / f"{base}_original_en.txt").write_text(chunk, encoding="utf-8")
                (debug_dir / f"{base}_context.txt").write_text(previous_context or "", encoding="utf-8")
                (debug_dir / f"{base}_llm_raw.txt").write_text(raw_text, encoding="utf-8")
                (debug_dir / f"{base}_final_pt.txt").write_text(parsed_clean, encoding="utf-8")
        except RuntimeError as exc:
            failed_chunks.add(idx)
            placeholder = f"[ERRO: chunk {idx} nao traduzido por timeout - revisar depois]"
            logger.warning(
                "Chunk trad-%d falhou apos %d tentativas; adicionando placeholder. Erro: %s",
                idx,
                cfg.max_retries,
                exc,
            )
            translated_chunks.append(placeholder)
            chunk_outputs[idx] = placeholder
            fallbacks += 1
        finally:
            previous_context = _extract_last_sentence(chunk)
            _write_progress()

    logger.info(
        "Resumo da traducao: total=%d sucesso=%d erro=%d",
        total_chunks,
        len(translated_ok),
        len(failed_chunks),
    )

    result = "\n\n".join(translated_chunks).strip()
    if not result:
        raise ValueError("Traducao resultou em texto vazio.")

    translated_paragraphs = [p for p in result.split("\n\n") if p.strip()]
    if len(translated_paragraphs) < len(paragraphs):
        logger.warning(
            "Paragrafos ausentes apos traducao: original=%d traduzido=%d",
            len(paragraphs),
            len(translated_paragraphs),
        )

    result, final_report = sanitize_text(result, logger=logger)
    log_report(final_report, logger, prefix="trad-final")
    try:
        version = (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        version = "unknown"
    report = {
        "mode": "translate",
        "input": source_slug or "",
        "total_chunks": total_chunks,
        "cache_hits": cache_hits,
        "fallbacks": fallbacks,
        "collapse_detected": collapse_detected,
        "duplicates_reused": duplicate_reuse,
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": version,
    }
    try:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.output_dir) / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    return result


def _call_with_retry(
    backend: LLMBackend,
    prompt: str,
    cfg: AppConfig,
    logger: logging.Logger,
    label: str,
) -> tuple[str, str]:
    """Chama backend com retry e sanitizacao agressiva."""
    delay = cfg.initial_backoff
    last_error: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            latency, response = timed(backend.generate, prompt)
            text, report = sanitize_text(response.text, logger=logger, fail_on_contamination=False)
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
            return response.text, text
        except Exception as exc:
            last_error = exc
            logger.warning("%s falhou (tentativa %d/%d): %s", label, attempt, cfg.max_retries, exc)
            if attempt < cfg.max_retries:
                time.sleep(delay)
                delay *= cfg.backoff_factor
    raise RuntimeError(f"{label} falhou apos {cfg.max_retries} tentativas: {last_error}")
