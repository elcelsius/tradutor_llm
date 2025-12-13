"""
Aplicacao de desquebrar usando o mesmo backend LLM do restante do pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import AppConfig
from .cache_utils import cache_exists, chunk_hash, load_cache, save_cache
from .llm_backend import LLMBackend
from .preprocess import paragraphs_from_text
from .utils import chunk_by_paragraphs, timed
import re


DESQUEBRAR_PROMPT = """
UNA APENAS AS QUEBRAS DE LINHA ERRADAS DO TEXTO ENTRE AS MARCAS ABAIXO.
NAO REESCREVA, NAO TRADUZA, NAO RESUMA, NAO ADICIONE NADA.
NAO TROQUE PALAVRAS, NAO MUDE PONTUACAO, NAO MUDE NENHUM TERMO.
RETORNE SOMENTE O TEXTO CORRIGIDO, SEM CABECALHOS OU COMENTARIOS.

TEXTO:
\"\"\"{chunk}\"\"\""""


@dataclass
class DesquebrarStats:
    total_chunks: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    blocks: list[dict] | None = None


def build_desquebrar_prompt(chunk: str) -> str:
    return DESQUEBRAR_PROMPT.format(chunk=chunk)


def desquebrar_text(
    text: str,
    cfg: AppConfig,
    logger: logging.Logger,
    backend: LLMBackend,
    chunk_chars: int | None = None,
) -> tuple[str, DesquebrarStats]:
    """
    Normaliza quebras de linha com LLM respeitando chunking seguro.

    Retorna (texto_desquebrado, stats).
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return "", DesquebrarStats(total_chunks=0, cache_hits=0, fallbacks=0, blocks=[])
    paragraphs = paragraphs_from_text(normalized)
    if len(paragraphs) <= 1:
        paragraphs = [ln.strip() for ln in normalized.splitlines() if ln.strip()]

    max_chars = chunk_chars or cfg.desquebrar_chunk_chars
    chunks = chunk_by_paragraphs(paragraphs, max_chars=max_chars, logger=logger, label="desquebrar")
    stats = DesquebrarStats(total_chunks=len(chunks), blocks=[])

    outputs: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        h = chunk_hash(chunk)
        from_cache = False
        if cache_exists("desquebrar", h):
            data = load_cache("desquebrar", h)
            meta_ok = False
            meta = data.get("metadata")
            expected = {
                "backend": getattr(backend, "backend", None),
                "model": getattr(backend, "model", None),
                "num_predict": getattr(backend, "num_predict", None),
                "temperature": getattr(backend, "temperature", None),
                "chunk_chars": max_chars,
                "repeat_penalty": getattr(backend, "repeat_penalty", None),
            }
            if isinstance(meta, dict):
                meta_ok = all(meta.get(k) == v for k, v in expected.items())
            if not meta_ok:
                logger.debug("Cache de desquebrar ignorado: assinatura diferente.")
            cached = data.get("final_output") if meta_ok else None
            if cached:
                logger.info("Reusando cache de desquebrar para bloco %d/%d", idx, len(chunks))
                outputs.append(cached)
                stats.cache_hits += 1
                from_cache = True

        if from_cache:
            stats.blocks.append(
                {
                    "chunk_index": idx,
                    "chars_in": len(chunk),
                    "chars_out": len(outputs[-1]) if outputs else 0,
                    "from_cache": True,
                    "fallback": False,
                }
            )
            continue

        prompt = build_desquebrar_prompt(chunk)
        try:
            latency, response = timed(backend.generate, prompt)
            cleaned = response.text.strip()
            if not cleaned:
                raise ValueError("Resposta vazia do desquebrar.")
            outputs.append(cleaned)
            stats.blocks.append(
                {
                    "chunk_index": idx,
                    "chars_in": len(chunk),
                    "chars_out": len(cleaned),
                    "latency": latency,
                    "from_cache": False,
                    "fallback": False,
                }
            )
            save_cache(
                "desquebrar",
                h,
                raw_output=response.text,
                final_output=cleaned,
                metadata={
                    "chunk_index": idx,
                    "mode": "desquebrar",
                    "model": getattr(backend, "model", None),
                    "backend": getattr(backend, "backend", None),
                    "num_predict": getattr(backend, "num_predict", None),
                    "temperature": getattr(backend, "temperature", None),
                    "chunk_chars": max_chars,
                    "repeat_penalty": getattr(backend, "repeat_penalty", None),
                },
            )
        except Exception as exc:  # pragma: no cover - network/LLM failure path
            logger.warning("Bloco %d do desquebrar falhou; mantendo texto original. Erro: %s", idx, exc)
            outputs.append(chunk)
            stats.fallbacks += 1
            stats.blocks.append(
                {
                    "chunk_index": idx,
                    "chars_in": len(chunk),
                    "chars_out": len(chunk),
                    "from_cache": False,
                    "fallback": True,
                    "error": str(exc),
                }
            )

    combined = "\n\n".join(outputs).strip()
    return combined, stats


def desquebrar_stats_to_dict(stats: DesquebrarStats | None, cfg: AppConfig) -> dict:
    if stats is None:
        return {}
    return {
        "total_chunks": stats.total_chunks,
        "cache_hits": stats.cache_hits,
        "fallbacks": stats.fallbacks,
        "blocks": stats.blocks or [],
        "effective_desquebrar_chunk_chars": cfg.desquebrar_chunk_chars,
        "backend": getattr(cfg, "desquebrar_backend", None),
        "model": getattr(cfg, "desquebrar_model", None),
    }


def normalize_md_paragraphs(md_text: str) -> str:
    """
    Normaliza parágrafos juntando linhas internas, preservando blocos especiais.
    """
    if not md_text:
        return md_text

    text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    normalized: list[str] = []
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            normalized.append(" ".join(buffer).strip())
            buffer = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if in_fence:
            normalized.append(raw_line)
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_buffer()
            in_fence = True
            fence_marker = stripped[:3]
            normalized.append(raw_line)
            continue

        if stripped == "":
            flush_buffer()
            normalized.append("")
            continue

        if (
            re.match(r"^#{1,6}\s", stripped)
            or re.match(r"^>\s", stripped)
            or re.match(r"^[-*+]\s", stripped)
            or re.match(r"^\d+\.\s", stripped)
        ):
            flush_buffer()
            normalized.append(stripped)
            continue

        if buffer:
            if buffer[-1].endswith("-"):
                buffer[-1] = buffer[-1][:-1]
                buffer.append(stripped.lstrip())
            else:
                buffer.append(stripped)
        else:
            buffer.append(stripped)

    flush_buffer()

    compact: list[str] = []
    prev_blank = False
    for ln in normalized:
        if ln == "":
            if not prev_blank:
                compact.append("")
            prev_blank = True
        else:
            compact.append(ln)
            prev_blank = False

    return "\n".join(compact).strip()
