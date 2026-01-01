"""
Aplicacao de desquebrar usando o mesmo backend LLM do restante do pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import re

from .config import AppConfig
from .cache_utils import cache_exists, chunk_hash, load_cache, save_cache, set_cache_base_dir
from .llm_backend import LLMBackend
from .preprocess import paragraphs_from_text
from .utils import chunk_by_paragraphs, timed
from .desquebrar_safe import safe_reflow


DESQUEBRAR_PROMPT = """
UNA APENAS AS QUEBRAS DE LINHA ERRADAS DO TEXTO ENTRE AS MARCAS ABAIXO.
NAO REESCREVA, NAO TRADUZA, NAO RESUMA, NAO ADICIONE NADA.
NAO TROQUE PALAVRAS, NAO MUDE PONTUACAO, NAO MUDE NENHUM TERMO.
RETORNE SOMENTE O TEXTO CORRIGIDO, SEM CABECALHOS OU COMENTARIOS.

TEXTO:
\"\"\"{chunk}\"\"\""""

ELLIPSIS_RE = re.compile(r"\.\.\.|…")
SAFE_DIALOGUE_BREAK_PATTERNS = (
    (re.compile(r"”{2,}\s*“"), "”\n\n“"),
    (re.compile(r"”\s*“"), "”\n\n“"),
)


@dataclass
class DesquebrarStats:
    total_chunks: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    dialogue_splits: int = 0
    blocks: list[dict] | None = None
    hyphen_linewrap_count: int = 0
    stutter_space_count: int = 0
    stray_quote_lines: int = 0


def _count_alnum(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


def _count_ellipses(text: str) -> int:
    return len(ELLIPSIS_RE.findall(text))


def _has_lonely_quote_line(text: str) -> bool:
    return any(line.strip() in {'"', "“", "”"} for line in text.splitlines())


def _count_quotes(text: str) -> int:
    return sum(1 for ch in text if ch in {'"', "“", "”"})


def _isolate_asterisks(text: str) -> str:
    lines = text.splitlines()
    new_lines: list[str] = []
    for i, ln in enumerate(lines):
        if ln.strip() == "***":
            if new_lines and new_lines[-1].strip() != "":
                new_lines.append("")
            new_lines.append("***")
            if i + 1 < len(lines) and lines[i + 1].strip() != "":
                new_lines.append("")
            continue
        new_lines.append(ln)
    return "\n".join(new_lines)


def _postprocess_output(orig: str, text: str) -> tuple[str, dict]:
    """
    Ajustes determinísticos pós-LLM para o desquebrar.
    """
    metrics = {"hyphen_linewrap_count": 0, "stutter_space_count": 0, "stray_quote_lines": 0}
    cleaned = text.strip()
    for _ in range(2):
        if cleaned.startswith('"""'):
            cleaned = cleaned[3:].lstrip()
        if cleaned.endswith('"""'):
            cleaned = cleaned[:-3].rstrip()
    lines = cleaned.splitlines()
    kept_lines: list[str] = []
    for ln in lines:
        if ln.strip() in {'"', "“", "”", "'''", '"""'}:
            metrics["stray_quote_lines"] += 1
            continue
        kept_lines.append(ln)
    cleaned = "\n".join(kept_lines)

    cleaned, hcount = re.subn(r"(\w)-\s*\n\s*(\w)", r"\1-\2", cleaned)
    metrics["hyphen_linewrap_count"] += hcount
    cleaned, scount = re.subn(r"\b([A-Za-zÀ-ÿ])-\s+([A-Za-zÀ-ÿ])", r"\1-\2", cleaned)
    metrics["stutter_space_count"] += scount

    cleaned = _isolate_asterisks(cleaned)
    return cleaned, metrics


def validate_desquebrar_output(orig: str, output: str) -> tuple[bool, list[str]]:
    """
    Valida a saida do desquebrar para garantir que nada foi corrompido.
    Retorna (ok, motivos_de_rejeicao).
    """
    reasons: list[str] = []
    if _has_lonely_quote_line(output):
        reasons.append("lonely_quote_line")
    if _count_ellipses(output) != _count_ellipses(orig):
        reasons.append("ellipsis_mismatch")
    orig_alnum = _count_alnum(orig)
    out_alnum = _count_alnum(output)
    if orig_alnum and out_alnum < orig_alnum * 0.99:
        reasons.append("alnum_loss")
    return not reasons, reasons


def deterministic_unbreak(text: str) -> str:
    """
    Reflow deterministico que apenas junta linhas do mesmo paragrafo.
    - Nao atravessa linhas em branco.
    - Mantem headings isolados (linhas que comecam com #).
    - Remove hifenizacao no fim de linha apenas quando a proxima comeca minuscula.
    """
    if not text:
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = normalized.split("\n\n")
    rebuilt: list[str] = []
    for para in paragraphs:
        if para.strip() == "":
            rebuilt.append("")
            continue
        lines = para.split("\n")
        acc = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if acc:
                    rebuilt.append(acc.strip())
                    acc = ""
                rebuilt.append(stripped)
                continue
            if acc:
                if acc.endswith("-") and stripped[:1].islower():
                    acc = acc[:-1] + stripped
                else:
                    acc = f"{acc} {stripped}"
            else:
                acc = stripped
        if acc or (not rebuilt or rebuilt[-1] != ""):
            rebuilt.append(acc.strip())
    return "\n\n".join(rebuilt).strip()


def normalize_dialogue_breaks_source_safe(text: str) -> tuple[str, dict]:
    """
    Variante segura que nao introduz linhas contendo apenas aspas retas.
    """
    if not text:
        return text, {"dialogue_splits": 0}
    cleaned = text
    total = 0
    for pattern, replacement in SAFE_DIALOGUE_BREAK_PATTERNS:
        cleaned, count = pattern.subn(replacement, cleaned)
        total += count
    return cleaned, {"dialogue_splits": total}


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
    set_cache_base_dir(cfg.output_dir)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return "", DesquebrarStats(total_chunks=0, cache_hits=0, fallbacks=0, blocks=[])
    paragraphs = paragraphs_from_text(normalized)
    if len(paragraphs) <= 1:
        paragraphs = [ln.strip() for ln in normalized.splitlines() if ln.strip()]

    max_chars = chunk_chars or cfg.desquebrar_chunk_chars
    chunks = chunk_by_paragraphs(paragraphs, max_chars=max_chars, logger=logger, label="desquebrar")
    total_chunks = len(chunks)
    stats = DesquebrarStats(total_chunks=total_chunks, blocks=[])

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
                logger.info("desq-%d/%d cache_hit", idx, total_chunks)
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
            cleaned = response.text
            cleaned, pp_metrics = _postprocess_output(chunk, cleaned)
            stats.hyphen_linewrap_count += pp_metrics.get("hyphen_linewrap_count", 0)
            stats.stutter_space_count += pp_metrics.get("stutter_space_count", 0)
            stats.stray_quote_lines += pp_metrics.get("stray_quote_lines", 0)
            if not cleaned.strip():
                raise ValueError("Resposta vazia do desquebrar.")
            in_quotes = _count_quotes(chunk)
            out_quotes = _count_quotes(cleaned)
            quote_inflation = in_quotes == 0 and out_quotes > 2 or (in_quotes > 0 and out_quotes > max(int(in_quotes * 1.5), in_quotes + 2))
            fallback_trigger = pp_metrics.get("stray_quote_lines", 0) > 0 or quote_inflation
            is_valid, reasons = validate_desquebrar_output(chunk, cleaned)
            if fallback_trigger:
                reasons = reasons or []
                reasons.append("quote_inflation" if quote_inflation else "stray_quote_lines")
                is_valid = False
            if not is_valid:
                fallback_text = safe_reflow(chunk)
                outputs.append(fallback_text)
                stats.fallbacks += 1
                reason_str = ",".join(reasons) or "invalid_output"
                logger.warning(
                    "desq-%d/%d invalid output; usando fallback deterministico (reasons=%s)",
                    idx,
                    total_chunks,
                    reason_str,
                )
                stats.blocks.append(
                    {
                        "chunk_index": idx,
                        "chars_in": len(chunk),
                        "chars_out": len(fallback_text),
                        "latency": latency,
                        "from_cache": False,
                        "fallback": True,
                        "fallback_reason": reason_str,
                    }
                )
                continue
            outputs.append(cleaned)
            logger.info("desq-%d/%d ok (%.2fs, %d chars)", idx, total_chunks, latency, len(cleaned))
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
            logger.warning("Bloco %d do desquebrar falhou; usando fallback deterministico. Erro: %s", idx, exc)
            fallback_text = deterministic_unbreak(chunk)
            outputs.append(fallback_text)
            stats.fallbacks += 1
            logger.info("desq-%d/%d fallback", idx, total_chunks)
            stats.blocks.append(
                {
                    "chunk_index": idx,
                    "chars_in": len(chunk),
                    "chars_out": len(fallback_text),
                    "from_cache": False,
                    "fallback": True,
                    "fallback_reason": "exception",
                    "error": str(exc),
                }
            )

    combined = "\n\n".join(outputs).strip()
    combined, norm_stats = normalize_dialogue_breaks_source_safe(combined)
    combined = normalize_wrapped_lines(combined)
    stats.dialogue_splits = norm_stats.get("dialogue_splits", 0)
    logger.info(
        "Desquebrar métricas: hyphen_linewrap=%d stutter_space=%d stray_quote_lines=%d",
        stats.hyphen_linewrap_count,
        stats.stutter_space_count,
        stats.stray_quote_lines,
    )
    return combined, stats


def normalize_wrapped_lines(text: str) -> str:
    """
    Junta quebras de linha erradas em narrativas curtas (sem mexer em falas/headers).
    Critérios de junção:
    - Linha não termina com pontuação final (.?!…:"”)
    - Próxima linha começa minúscula
    - Nenhuma das duas começa com aspas/travessão/heading
    """
    if not text:
        return text
    lines = text.split("\n")
    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            i += 1
            continue
        if stripped.startswith(('"', "“", "”", "-", "—", "#")):
            new_lines.append(line)
            i += 1
            continue
        ends_ok = stripped.endswith((".", "?", "!", "…", ":", "”", '"'))
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            nxt_stripped = nxt.strip()
            starts_lower = bool(nxt_stripped) and nxt_stripped[:1].islower()
            nxt_block = nxt_stripped.startswith(('"', "“", "”", "-", "—", "#"))
            if (not ends_ok) and starts_lower and not nxt_block:
                merged = f"{stripped} {nxt_stripped}"
                new_lines.append(merged)
                i += 2
                continue
        new_lines.append(line)
        i += 1
    return "\n".join(new_lines)


def desquebrar_stats_to_dict(stats: DesquebrarStats | None, cfg: AppConfig) -> dict:
    if stats is None:
        return {}
    return {
        "total_chunks": stats.total_chunks,
        "cache_hits": stats.cache_hits,
        "fallbacks": stats.fallbacks,
        "dialogue_splits": stats.dialogue_splits,
        "hyphen_linewrap_count": stats.hyphen_linewrap_count,
        "stutter_space_count": stats.stutter_space_count,
        "stray_quote_lines": stats.stray_quote_lines,
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
