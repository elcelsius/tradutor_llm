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
    """Prompt de refine com foco em fluidez sem alterar sentido."""
    return f"""
Você é um revisor literário profissional em PT-BR.
- Melhore fluidez, coesão e pontuação mantendo sentido.
- Não resuma, não invente, não troque nomes.
- Preserve Markdown básico (títulos, negrito) se houver.
- Não adicione comentários nem instruções.
- Não use <think>.

Texto para refinar:
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
        text = _call_with_retry(
            backend=backend,
            prompt=prompt,
            cfg=cfg,
            logger=logger,
            label=f"ref-{index}/{total}-{c_idx}/{len(chunks)}",
        )
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
