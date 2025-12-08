"""
Pipeline de tradução em lotes.
"""

from __future__ import annotations

import logging
import time
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
Você é um tradutor literário profissional especializado em light novels e webnovels.
Idioma de origem: inglês (funcione mesmo se vier outra língua).
Idioma de destino: português brasileiro.
Regras:
- Mantenha sentido, enredo, tom emocional, ritmo da cena e personalidade dos personagens.
- Preserve nomes próprios, termos importantes e a formatação básica, incluindo quebras de parágrafo.
- Em diálogos, faça soar como fala natural em PT-BR, sem travar e sem literalidade robótica.
- Piadas e expressões idiomáticas podem ser adaptadas para algo natural em PT-BR, mas não adicione fatos novos e não remova o efeito cômico.
- Em conteúdo pesado, traduza de forma fiel, sem censura ou suavização.
- Se o trecho estiver incompleto, traduza apenas o que recebeu, sem inventar continuação.
- Proibido resumir, comentar, explicar piadas ou falar como modelo.
- Não use <think> nem qualquer marcação fora do texto original.

Texto original:
\"\"\"{chunk}\"\"\"
"""


def translate_document(
    pdf_text: str,
    backend: LLMBackend,
    cfg: AppConfig,
    logger: logging.Logger,
) -> str:
    """
    Executa pré-processamento, chunking e tradução por lotes com sanitização.
    """
    clean = preprocess_text(pdf_text, logger)
    paragraphs = paragraphs_from_text(clean)
    chunks = chunk_for_translation(paragraphs, max_chars=cfg.translate_chunk_chars, logger=logger)
    logger.info("Iniciando tradução: %d chunks", len(chunks))

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
            text, report = sanitize_text(response.text, logger=logger)
            log_report(report, logger, prefix=label)
            logger.info("%s ok (%.2fs, %d chars)", label, latency, len(text))
            return text
        except Exception as exc:
            last_error = exc
            logger.warning("%s falhou (tentativa %d/%d): %s", label, attempt, cfg.max_retries, exc)
            time.sleep(delay)
            delay *= cfg.backoff_factor
    raise RuntimeError(f"{label} falhou após {cfg.max_retries} tentativas: {last_error}")
