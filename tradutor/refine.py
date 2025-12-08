"""
Refinamento capítulo a capítulo de arquivos Markdown.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import List, Tuple

from .config import AppConfig
from .llm_backend import LLMBackend
from .preprocess import chunk_for_refine, paragraphs_from_text
from .sanitizer import sanitize_text, log_report
from .utils import read_text, timed, write_text


def build_refine_prompt(section: str) -> str:
    return f"""
Você é um REVISOR LITERÁRIO PROFISSIONAL especializado em novels adultas, dark fantasy e narrativa intensa.

Sua função é revisar o texto abaixo MELHORANDO:

– fluidez,  
– coesão,  
– naturalidade literária,  
– ortografia,  
– concordância verbal e nominal,  
– ritmo e legibilidade.

REGRAS ABSOLUTAS:

1) NÃO apagar, cortar, omitir ou resumir nenhuma informação do texto original.  
   Cada frase importante deve continuar existindo.  
   Se algo parecer ambíguo, mantenha.

2) NÃO adicionar conteúdo, frases novas, explicações, notas, análises ou qualquer texto que não esteja no original.

3) NÃO suavizar xingamentos, insultos, blasfêmias, agressões, violência verbal.  
   O tom adulto deve ser preservado integralmente.

4) Corrigir erros de português e frases quebradas:
   – “E essa é a fim dela.” deve ser corrigido para algo natural como:
     “E é aí que tudo termina.” ou “E esse é o fim disso.”

5) Corrigir concordância de gênero do narrador:
   – Se o narrador é masculino, use “pronto”, “determinado”, etc.

6) Corrigir plural indevido:
   – Não transformar singular em plural:
     “você” NÃO deve virar “vocês”.

7) Naturalizar expressões rígidas:
   – “através da floresta” → “pela floresta”.

8) Manter estrutura e quebras de parágrafo exatamente como no original.

9) A resposta deve ser APENAS o texto revisado.

Texto a revisar:
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
) -> str:
    paragraphs = paragraphs_from_text(body)
    chunks = chunk_for_refine(paragraphs, max_chars=cfg.refine_chunk_chars, logger=logger)
    logger.info("Refinando seção %s (%d chunks)", title or f"#{index}", len(chunks))
    refined_parts: List[str] = []

    for c_idx, chunk in enumerate(chunks, start=1):
        prompt = build_refine_prompt(chunk)
        logger.debug("Refinando seção com %d caracteres...", len(chunk))
        text = _call_with_retry(
            backend=backend,
            prompt=prompt,
            cfg=cfg,
            logger=logger,
            label=f"ref-{index}/{total}-{c_idx}/{len(chunks)}",
        )
        if len(text.strip()) < len(chunk.strip()) * 0.80:
            logger.warning(
                "Refinador devolveu texto menor do que deveria; mantendo texto original (chunk %d/%d da seção %s).",
                c_idx,
                len(chunks),
                title or f"#{index}",
            )
            text = chunk
        logger.debug("Seção refinada com %d caracteres.", len(text))
        refined_parts.append(text)

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
) -> None:
    md_text = read_text(input_path)
    sections = split_markdown_sections(md_text)
    logger.info("Arquivo %s: %d seções detectadas", input_path.name, len(sections))

    refined_sections: List[str] = []
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
            )
        )

    final_md = "\n\n".join(refined_sections).strip()
    if not final_md:
        raise ValueError(f"Refine produziu texto vazio para {input_path}")

    final_md, final_report = sanitize_text(final_md, logger=logger)
    log_report(final_report, logger, prefix="ref-final")

    write_text(output_path, final_md)
    logger.info("Refine concluído: %s", output_path.name)


def _call_with_retry(
    backend: LLMBackend,
    prompt: str,
    cfg: AppConfig,
    logger: logging.Logger,
    label: str,
) -> str:
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
