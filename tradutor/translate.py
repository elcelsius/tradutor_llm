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
Você é um tradutor literário profissional especializado em novels, light novels e webnovels.
Sua tarefa é traduzir o texto abaixo do INGLÊS para o PORTUGUÊS BRASILEIRO de forma absolutamente fiel.

Regras OBRIGATÓRIAS (leia atentamente):

1. **Nenhuma frase do original pode ser omitida. NUNCA.**  
   Para cada frase existente no texto original, você deve produzir uma frase correspondente em português.

2. **É proibido resumir, parafrasear, interpretar, suavizar ou alterar o impacto emocional.**  
   Nada de "resumir a ideia". Tudo deve aparecer claramente na tradução.

3. **Não altere intensidade emocional, violência, ameaças, votos de vingança, hostilidade ou dramaticidade.**  
   Exemplo:  
   - “I won’t stop until you’re dead.” deve aparecer explicitamente em PT-BR com o mesmo peso.  
   - Não transformar frases fortes em versões neutras ou genéricas.

4. **Mantenha a ordem, estrutura e segmentação do texto.**  
   - Respeite todos os parágrafos.  
   - Não una parágrafos diferentes.  
   - Não quebre frases que não estão quebradas.

5. **Preserve TODAS as informações do texto.**  
   - Nomes, emoções, metáforas, termos mágicos, títulos, referências culturais.  
   - Nada deve desaparecer.

6. **Preserve e traduza fielmente todos os DIÁLOGOS.**  
   - Se houver falas, todas devem aparecer.  
   - Não pule nenhuma linha de diálogo.  
   - Não resuma falas.  
   - Não altere o conteúdo dito pelos personagens.

7. **Adaptações são permitidas APENAS para naturalidade do português brasileiro**, NÃO para alterar o conteúdo.  
   - Ajustes de ordem das palavras são OK.  
   - Suavizar, retirar ou inventar informações NÃO é permitido.

8. **NÃO ADICIONE NADA.**  
   - Não explique.  
   - Não comente.  
   - Não coloque notas.  
   - Não insira pensamentos extras.  
   - Sua resposta deve conter APENAS a tradução, nada mais.

Quando encontrar a expressão **“Foul Goddess…”**, traduza como **“Deusa desgraçada…”**. Não use versões suaves como “Deusa Maldosa”, “Deusa Malvada” ou equivalentes. O tom deve ser agressivo, carregado de rancor e hostilidade.

9. **NÃO USE `<think>` ou qualquer forma de raciocínio oculto.**

10. **O resultado deve ser um texto literário natural, fluente e emocionalmente fiel ao original.**

---

TEXTO ORIGINAL A SER TRADUZIDO:
\"\"\"
{chunk}
\"\"\"

APENAS produza a tradução completa e fiel do texto — sem comentários, sem introduções, sem explicações.
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
