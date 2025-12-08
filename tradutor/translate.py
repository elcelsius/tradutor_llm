"""
Pipeline de tradução em lotes.
"""

from __future__ import annotations

import logging
import time
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

5) Preservar o número original:
   – Não transformar “you” singular em “vocês”.

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
    for idx, chunk in enumerate(chunks, start=1):
        prompt = build_translation_prompt(chunk)
        text = _call_with_retry(
            backend=backend,
            prompt=prompt,
            cfg=cfg,
            logger=logger,
            label=f"trad-{idx}/{len(chunks)}",
        )
        translated_chunks.append(text)

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
