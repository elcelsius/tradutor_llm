"""
Pipeline de tradução em lotes.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from .config import AppConfig
from .llm_backend import LLMBackend
from .preprocess import (
    chunk_for_translation,
    paragraphs_from_text,
    preprocess_text,
)
from .sanitizer import sanitize_text, log_report
from .utils import timed


def build_translation_prompt(chunk: str) -> str:
    """Prompt especializado para tradução literária de novels para PT-BR."""
    return f"""
Você é um TRADUTOR LITERÁRIO PROFISSIONAL especializado em novels adultas, com violência, vingança, blasfêmias e narrativa emocional intensa.

Seu trabalho é traduzir do INGLÊS para o PORTUGUÊS BRASILEIRO com:

– fidelidade total,  
– preservação do tom adulto,  
– zero suavização,  
– zero censura.

REGRAS:

1) NÃO adicionar frases novas.  
   NÃO inventar interjeições como “Apertem!”, “Vamos!”, “起來！” e similares.

2) NÃO deixar nada em inglês, exceto nomes próprios.

3) NÃO suavizar agressões, insultos, xingamentos ou blasfêmias.
   Pode usar termos fortes como:
   “desgraçada”, “imunda”, “nojenta”, “maldito”, etc.

4) Preservar o gênero do narrador:
   – “When I’m ready” → “Quando eu estiver pronto”.

5) Preservar o número original e evitar plural indevido:
   - Não transformar "you" singular em "vocês" (não criar plural sem motivo).

6) Preservar estrutura de parágrafos e diálogos.

7) Responder APENAS com a tradução.

Texto original:
\"\"\"{chunk}\"\"\"
"""


def translate_document(
    pdf_text: str,
    backend: LLMBackend,
    cfg: AppConfig,
    logger: logging.Logger,
    source_slug: str | None = None,
    progress_path: Path | None = None,
    resume_manifest: dict | None = None,
) -> str:
    """
    Executa pré-processamento, chunking e tradução por lotes com sanitização.
    """
    clean = preprocess_text(pdf_text, logger)
    paragraphs = paragraphs_from_text(clean)
    chunks = chunk_for_translation(paragraphs, max_chars=cfg.translate_chunk_chars, logger=logger)
    logger.info("Iniciando tradução: %d chunks", len(chunks))

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
                    "Manifesto marca chunk %s como traduzido, mas não há conteúdo salvo; retraduzindo.",
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

    for idx, chunk in enumerate(chunks, start=1):
        if idx in translated_ok and idx in chunk_outputs:
            logger.info("Reusando tradução salva para chunk trad-%d/%d", idx, total_chunks)
            translated_chunks.append(chunk_outputs[idx])
            _write_progress()
            continue

        prompt = build_translation_prompt(chunk)
        try:
            text = _call_with_retry(
                backend=backend,
                prompt=prompt,
                cfg=cfg,
                logger=logger,
                label=f"trad-{idx}/{len(chunks)}",
            )
            translated_chunks.append(text)
            translated_ok.add(idx)
            failed_chunks.discard(idx)
            chunk_outputs[idx] = text
        except RuntimeError as exc:
            failed_chunks.add(idx)
            placeholder = f"[ERRO: chunk {idx} não traduzido por timeout - revisar depois]"
            logger.warning(
                "Chunk trad-%d falhou após %d tentativas; adicionando placeholder. Erro: %s",
                idx,
                cfg.max_retries,
                exc,
            )
            translated_chunks.append(placeholder)
            chunk_outputs[idx] = placeholder
        finally:
            _write_progress()

    logger.info(
        "Resumo da tradução: total=%d sucesso=%d erro=%d",
        total_chunks,
        len(translated_ok),
        len(failed_chunks),
    )

    result = "\n\n".join(translated_chunks).strip()
    if not result:
        raise ValueError("Tradução resultou em texto vazio.")

    translated_paragraphs = [p for p in result.split("\n\n") if p.strip()]
    if len(translated_paragraphs) < len(paragraphs):
        logger.warning(
            "Parágrafos ausentes após tradução: original=%d traduzido=%d",
            len(paragraphs),
            len(translated_paragraphs),
        )

    # Sanitização final
    result, final_report = sanitize_text(result, logger=logger)
    log_report(final_report, logger, prefix="trad-final")
    return result


def _call_with_retry(
    backend: LLMBackend,
    prompt: str,
    cfg: AppConfig,
    logger: logging.Logger,
    label: str,
) -> str:
    """Chama backend com retry e sanitização agressiva."""
    delay = cfg.initial_backoff
    last_error: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            latency, response = timed(backend.generate, prompt)
            text, report = sanitize_text(response.text, logger=logger, fail_on_contamination=False)
            log_report(report, logger, prefix=label)
            if not text.strip():
                raise ValueError("Texto vazio após sanitização.")
            if report.contamination_detected:
                logger.warning(
                    "%s: contaminação detectada; texto limpo será usado (%d chars)",
                    label,
                    len(text),
                )
            logger.info("%s ok (%.2fs, %d chars)", label, latency, len(text))
            return text
        except Exception as exc:
            last_error = exc
            logger.warning("%s falhou (tentativa %d/%d): %s", label, attempt, cfg.max_retries, exc)
            time.sleep(delay)
            delay *= cfg.backoff_factor
    raise RuntimeError(f"{label} falhou após {cfg.max_retries} tentativas: {last_error}")
