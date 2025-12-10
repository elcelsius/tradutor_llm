"""
Refinamento capítulo a capítulo de arquivos Markdown.
"""

from __future__ import annotations

import json
import logging
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
    format_glossary_for_prompt,
    parse_glossary_suggestions,
    save_dynamic_glossary,
    split_refined_and_suggestions,
)
from .llm_backend import LLMBackend
from .preprocess import chunk_for_refine, paragraphs_from_text
from .sanitizer import sanitize_refine_output
from .utils import read_text, timed, write_text


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
    glossary_text = ""
    if glossary_enabled:
        block = glossary_block.strip() if glossary_block else ""
        glossary_lines = [
            "GLOSSÁRIO E TERMINOLOGIA (obrigatório):",
            "Obedeça estritamente ao glossário para nomes próprios, títulos, lugares e termos especiais.",
            "Se identificar termos novos importantes, sugira entradas ao final da resposta.",
        ]
        if block:
            glossary_lines.append("")
            glossary_lines.append(block)
        glossary_lines.extend(
            [
                "",
                "FORMATO DE SAÍDA:",
                "1) Texto refinado completo.",
                "2) Depois, bloco de sugestões de glossário delimitado exatamente por:",
                GLOSSARIO_SUGERIDO_INICIO,
                "key: ...",
                "pt: ...",
                "category: ...",
                "notes: ...",
                "---",
                GLOSSARIO_SUGERIDO_FIM,
                "Se não houver novos termos, deixe o bloco vazio ou escreva 'nenhum'.",
            ]
        )
        glossary_text = "\n".join(glossary_lines)

    return f"""
Você é um REVISOR LITERÁRIO PROFISSIONAL especializado em romances adultos, dark fantasy e narrativa intensa.

Sua tarefa é REVISAR o texto abaixo (já em português brasileiro), elevando-o para um registro MAIS LITERÁRIO, sem perder o tom emocional, a agressividade ou a personalidade do narrador.

FOCO PRINCIPAL:
- Melhorar fluidez, coesão e ritmo.
- Corrigir ortografia, pontuação e concordância verbal/nominal.
- Tornar o texto mais natural e elegante para publicação, em estilo de romance adulto.

REGRA DE ESTILO (OPÇÃO B):
- Reduza gírias e marcas de oralidade quando elas NÃO forem essenciais.
  Exemplos:
    - "tá" → "está"
    - "tô" → "estou"
    - "a gente" → "nós" (quando fizer sentido no contexto)
- Mantenha expressões informais APENAS quando forem claramente parte da voz do personagem ou do efeito cômico/dramático pretendido.
- Preserve o tom adulto, o sarcasmo e a ironia.

REGRAS ABSOLUTAS:

1) NÃO apagar, cortar, resumir nem omitir informações.
   Todas as frases relevantes devem continuar existindo no texto revisado.

2) NÃO adicionar explicações, comentários, análises, notas de rodapé ou frases novas que mudem o conteúdo.

3) NÃO suavizar xingamentos, insultos, blasfêmias ou ameaças.
   Se o texto ofende uma deusa, mantém a ofensa em grau equivalente (como "desgraçada", "imunda", "nojenta", etc.).

4) Corrigir frases estruturalmente erradas, mantendo o sentido.
   Exemplo:
   - "E essa é a fim dela." → "E é assim que tudo termina." ou "E esse é o fim disso."

5) Manter o gênero e o número corretos:
   - Se o narrador é masculino, use "pronto" (não "pronta").
   - Não transformar "você" singular em "vocês" sem motivo.

6) Manter as quebras de parágrafo.
   A estrutura de parágrafos deve ser igual à do texto original.

7) Não retornar instruções, explicações ou qualquer coisa além do texto revisado.
   A resposta deve ser APENAS o texto final revisado.

{glossary_text}

Texto para revisão:
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
) -> str:
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
            response_text = _call_with_retry(
                backend=backend,
                prompt=prompt,
                cfg=cfg,
                logger=logger,
                label=f"ref-{index}/{total}-{c_idx}/{len(chunks)}",
            )
            refined_text = response_text
            if glossary_state:
                refined_text, suggestion_block = split_refined_and_suggestions(response_text)
                suggestions = parse_glossary_suggestions(suggestion_block or "")
                if suggestions:
                    updated = apply_suggestions_to_state(glossary_state, suggestions, logger)
                    if updated:
                        save_dynamic_glossary(glossary_state, logger)
                        glossary_block = format_glossary_for_prompt(
                            glossary_state.combined_index,
                            glossary_prompt_limit,
                        )

            if len(refined_text.strip()) < len(chunk.strip()) * 0.80:
                logger.warning(
                    "Refinador devolveu texto menor do que deveria; mantendo texto original (chunk %d/%d da seção %s).",
                    c_idx,
                    len(chunks),
                    title or f"#{index}",
                )
                refined_text = chunk
            logger.debug("Seção refinada com %d caracteres.", len(refined_text))
            refined_parts.append(refined_text)
            if stats:
                stats.success_blocks += 1
            if progress:
                progress.refined_blocks.add(block_idx)
                progress.error_blocks.discard(block_idx)
                progress.chunk_outputs[block_idx] = refined_text
        except RuntimeError as exc:
            placeholder = f"<!-- ERRO: refine do bloco {c_idx} falhou por timeout – revisar manualmente -->"
            logger.warning(
                "Chunk ref-%d/%d-%d/%d falhou após %d tentativas; adicionando placeholder. Erro: %s",
                index,
                total,
                c_idx,
                len(chunks),
                cfg.max_retries,
                exc,
            )
            refined_parts.append(placeholder)
            if stats:
                stats.error_blocks += 1
            if progress:
                progress.error_blocks.add(block_idx)
                progress.chunk_outputs[block_idx] = placeholder
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
) -> None:
    md_text = read_text(input_path)
    if normalize_paragraphs:
        md_text = normalize_md_paragraphs(md_text)
    sections = split_markdown_sections(md_text)
    logger.info("Arquivo %s: %d seções detectadas", input_path.name, len(sections))
    stats = RefineStats()

    # Pré-computa total de blocos para progress
    total_blocks = 0
    for _, body in sections:
        paragraphs = paragraphs_from_text(body)
        chunks = chunk_for_refine(paragraphs, max_chars=cfg.refine_chunk_chars, logger=logger)
        total_blocks += len(chunks)

    if progress_path is None:
        progress_path = output_path.with_name(f"{output_path.stem}_progress.json")

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
                )
            )

    final_md = "\n\n".join(refined_sections).strip()
    if not final_md:
        raise ValueError(f"Refine produziu texto vazio para {input_path}")

    final_md = sanitize_refine_output(final_md)

    write_text(output_path, final_md)
    if glossary_state:
        save_dynamic_glossary(glossary_state, logger)
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
) -> str:
    delay = cfg.initial_backoff
    last_error: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            latency, response = timed(backend.generate, prompt)
            text = sanitize_refine_output(response.text)
            if not text.strip():
                raise ValueError("Texto vazio após sanitização do refine.")
            logger.info("%s ok (%.2fs, %d chars)", label, latency, len(text))
            return text
        except Exception as exc:
            last_error = exc
            logger.warning("%s falhou (tentativa %d/%d): %s", label, attempt, cfg.max_retries, exc)
            time.sleep(delay)
            delay *= cfg.backoff_factor
    raise RuntimeError(f"{label} falhou após {cfg.max_retries} tentativas: {last_error}")
