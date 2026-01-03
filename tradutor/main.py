"""
CLI principal para tradução e refine.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import AppConfig, ensure_paths, load_config
from .debug_run import DebugRunWriter
from .glossary_utils import build_glossary_state, format_manual_pairs_for_translation
from .llm_backend import LLMBackend
from .pdf_export import markdown_to_pdf
from .pdf_reader import extract_pdf_text
from .advanced_preprocess import clean_text as advanced_clean
from .preprocess import preprocess_text, strip_front_matter
from .refine import refine_markdown_file, refine_prompt_fingerprint
from .postprocess import final_pt_postprocess
from .translate import translate_document, translation_prompt_fingerprint
from .desquebrar import desquebrar_text, desquebrar_stats_to_dict, normalize_md_paragraphs
from .desquebrar_safe import desquebrar_safe
from .utils import setup_logging, write_text, read_text
from .structure_normalizer import normalize_structure
from .editor import editor_pipeline
from .pdf import convert_markdown_to_pdf
from .cache_utils import clear_cache, set_cache_base_dir


def build_parser(cfg: AppConfig) -> argparse.ArgumentParser:
    """Constroi o parser de argumentos com subcomandos traduz/refina."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--debug",
        action="store_true",
        help="Ativa logs detalhados e artefatos intermediarios.",
    )  # permite --debug antes ou depois do subcomando
    common.add_argument(
        "--request-timeout",
        type=int,
        default=cfg.request_timeout,
        help="Timeout (s) por chamada de modelo (Ollama/Gemini).",
    )
    parser = argparse.ArgumentParser(
        description="Tradutor e refinador de PDFs com LLMs.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcomando: traduzir
    t = sub.add_parser(
        "traduz",
        parents=[common],
        help="Traduz PDFs da pasta data/ (ou um arquivo especifico).",
    )
    t.add_argument("--input", type=str, help="PDF especifico para traduzir.")
    t.add_argument("--backend", type=str, choices=["ollama", "gemini"], default=cfg.translate_backend)
    t.add_argument("--model", type=str, default=cfg.translate_model)
    t.add_argument(
        "--num-predict",
        type=int,
        default=cfg.translate_num_predict,
        help="Limite de tokens gerados por chunk (Ollama).",
    )
    t.add_argument("--no-refine", action="store_true", help="Não executar refine após traduzir.")
    t.add_argument(
        "--desquebrar-mode",
        dest="desquebrar_mode",
        choices=["llm", "safe"],
        default=cfg.desquebrar_mode,
        help="Modo do desquebrar: llm usa LLM; safe usa desquebrar_safe sem LLM, preservando layout (padrao: config).",
    )
    t.add_argument(
        "--refine-mode",
        dest="desquebrar_mode",
        choices=["llm", "safe"],
        default=cfg.desquebrar_mode,
        help=argparse.SUPPRESS,
    )
    t.add_argument(
        "--resume",
        action="store_true",
        help="Retoma traducao usando manifesto de progresso existente (se houver).",
    )
    t.add_argument(
        "--use-glossary",
        action="store_true",
        help="Ativa uso do glossario manual durante a traducao (EN->PT).",
    )
    t.add_argument(
        "--manual-glossary",
        type=str,
        help="Arquivo JSON de glossario manual para a traducao (padrao: glossario/glossario_manual.json).",
    )
    t.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Numero de workers paralelos (traducao). Contexto mantem ordem; valores >1 sao ajustados se necessario.",
    )
    t.add_argument(
        "--preprocess-advanced",
        action="store_true",
        help="Ativa pre-processamento avancado opcional antes da traducao/refine.",
    )
    t.add_argument(
        "--skip-front-matter",
        dest="skip_front_matter",
        action=argparse.BooleanOptionalAction,
        default=cfg.skip_front_matter,
        help="Remove automaticamente o front-matter antes do Prologue/Chapter 1 (padrao: config).",
    )
    t.add_argument(
        "--clear-cache",
        choices=["all", "translate", "refine", "desquebrar"],
        help="Limpa caches antes de traduzir (respeita output_dir da config).",
    )
    t.add_argument(
        "--split-by-sections",
        dest="split_by_sections",
        action=argparse.BooleanOptionalAction,
        default=cfg.split_by_sections,
        help="Divide o texto em secoes (Prologue/Chapter/Epilogue) e traduz por secao (padrao: config).",
    )
    t.add_argument(
        "--translate-allow-adaptation",
        action=argparse.BooleanOptionalAction,
        default=cfg.translate_allow_adaptation,
        help="Permite exemplos de adaptacao de piadas/trocadilhos no prompt de traducao (padrao: config).",
    )
    t.add_argument(
        "--cleanup-before-refine",
        choices=["off", "auto", "on"],
        default=None,
        help="Controle de limpeza deterministica antes do refine: off, auto, on.",
    )
    t.add_argument(
        "--use-desquebrar",
        action=argparse.BooleanOptionalAction,
        default=cfg.use_desquebrar,
        help="Aplica desquebrar antes de traduzir (padrao: true para PDF). Desative com --no-use-desquebrar.",
    )
    t.add_argument(
        "--desquebrar-backend",
        type=str,
        choices=["ollama", "gemini"],
        default=cfg.desquebrar_backend,
        help="Backend LLM para desquebrar (padrao: ollama).",
    )
    t.add_argument(
        "--desquebrar-model",
        type=str,
        default=cfg.desquebrar_model,
        help="Modelo LLM para desquebrar.",
    )
    t.add_argument(
        "--desquebrar-temperature",
        type=float,
        default=cfg.desquebrar_temperature,
        help="Temperatura do desquebrar (padrao: 0.08).",
    )
    t.add_argument(
        "--desquebrar-chunk-chars",
        type=int,
        default=cfg.desquebrar_chunk_chars,
        help="Tamanho alvo (chars) por chunk no desquebrar.",
    )
    t.add_argument(
        "--desquebrar-num-predict",
        type=int,
        default=cfg.desquebrar_num_predict,
        help="Limite de tokens no desquebrar (Ollama).",
    )
    t.add_argument(
        "--desquebrar-repeat-penalty",
        type=float,
        default=cfg.desquebrar_repeat_penalty,
        help="Repeat penalty no desquebrar (Ollama).",
    )
    t.add_argument(
        "--debug-chunks",
        action="store_true",
        help="Ativa debug detalhado por chunk (JSONL) na traducao.",
    )
    t.add_argument(
        "--fail-on-chunk-error",
        action="store_true",
        help="Abortar tradução se qualquer chunk falhar (default: continua com placeholders).",
    )
    t.add_argument(
        "--pdf-enabled",
        action=argparse.BooleanOptionalAction,
        default=cfg.pdf_enabled,
        help="Gera PDF automaticamente após o refine (padrão: config).",
    )

    # Subcomando: traduzir Markdown/texto já desquebrado
    tm = sub.add_parser(
        "traduz-md",
        parents=[common],
        help="Traduz um arquivo .md/.txt já desquebrado (pula extração de PDF).",
    )
    tm.add_argument("--input", type=str, required=True, help="Arquivo .md ou .txt já desquebrado para traduzir.")
    tm.add_argument("--backend", type=str, choices=["ollama", "gemini"], default=cfg.translate_backend)
    tm.add_argument("--model", type=str, default=cfg.translate_model)
    tm.add_argument(
        "--num-predict",
        type=int,
        default=cfg.translate_num_predict,
        help="Limite de tokens gerados por chunk (Ollama).",
    )
    tm.add_argument("--no-refine", action="store_true", help="Não executar refine após traduzir.")
    tm.add_argument(
        "--resume",
        action="store_true",
        help="Retoma tradução usando manifesto de progresso existente (se houver).",
    )
    tm.add_argument(
        "--use-glossary",
        action="store_true",
        help="Ativa uso do glossario manual durante a traducao (EN->PT).",
    )
    tm.add_argument(
        "--manual-glossary",
        type=str,
        help="Arquivo JSON de glossario manual para a traducao (padrao: glossario/glossario_manual.json).",
    )
    tm.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Numero de workers paralelos (traducao). Contexto mantem ordem; valores >1 sao ajustados se necessario.",
    )
    tm.add_argument(
        "--preprocess-advanced",
        action="store_true",
        help="Ativa limpeza extra antes de traduzir o markdown/texto.",
    )
    tm.add_argument(
        "--normalize-paragraphs",
        action="store_true",
        help="Normaliza paragrafos do Markdown antes de traduzir.",
    )
    tm.add_argument(
        "--clear-cache",
        choices=["all", "translate", "refine", "desquebrar"],
        help="Limpa caches antes de traduzir (respeita output_dir da config).",
    )
    tm.add_argument(
        "--split-by-sections",
        dest="split_by_sections",
        action=argparse.BooleanOptionalAction,
        default=cfg.split_by_sections,
        help="Divide o texto em secoes (Prologue/Chapter/Epilogue) e traduz por secao (padrao: config).",
    )
    tm.add_argument(
        "--translate-allow-adaptation",
        action=argparse.BooleanOptionalAction,
        default=cfg.translate_allow_adaptation,
        help="Permite exemplos de adaptacao de piadas/trocadilhos no prompt de traducao (padrao: config).",
    )
    tm.add_argument(
        "--cleanup-before-refine",
        choices=["off", "auto", "on"],
        default=None,
        help="Controle de limpeza deterministica antes do refine: off, auto, on.",
    )
    tm.add_argument(
        "--debug-chunks",
        action="store_true",
        help="Ativa debug detalhado por chunk (JSONL) na traducao.",
    )
    tm.add_argument(
        "--fail-on-chunk-error",
        action="store_true",
        help="Abortar tradu‡Æo se qualquer chunk falhar (default: continua com placeholders).",
    )
    tm.add_argument(
        "--pdf-enabled",
        action=argparse.BooleanOptionalAction,
        default=cfg.pdf_enabled,
        help="Gera PDF automaticamente ap¢s o refine (padrÆo: config).",
    )

    # Subcomando: refinar
    r = sub.add_parser(
        "refina",
        parents=[common],
        help="Refina arquivos *_pt.md na pasta saida/ ou um arquivo especifico.",
    )
    r.add_argument("--input", type=str, help="Arquivo especifico para refinar (ex.: saida/xxx_pt.md).")
    r.add_argument("--backend", type=str, choices=["ollama", "gemini"], default=cfg.refine_backend)
    r.add_argument("--model", type=str, default=cfg.refine_model)
    r.add_argument(
        "--num-predict",
        type=int,
        default=cfg.refine_num_predict,
        help="Limite de tokens gerados por chunk no refine (Ollama).",
    )
    r.add_argument(
        "--desquebrar-mode",
        dest="desquebrar_mode",
        choices=["llm", "safe"],
        default=cfg.desquebrar_mode,
        help="Compatibilidade: controla apenas o desquebrar (safe usa desquebrar_safe). No comando refina não altera o fluxo.",
    )
    r.add_argument(
        "--refine-mode",
        dest="desquebrar_mode",
        choices=["llm", "safe"],
        default=cfg.desquebrar_mode,
        help=argparse.SUPPRESS,
    )
    r.add_argument(
        "--resume",
        action="store_true",
        help="Retoma refine usando manifesto de progresso existente (se houver).",
    )
    r.add_argument(
        "--clear-cache",
        choices=["all", "translate", "refine", "desquebrar"],
        help="Limpa caches antes de refinar (respeita output_dir da config).",
    )
    r.add_argument(
        "--normalize-paragraphs",
        action="store_true",
        help="Normaliza paragrafos do .md antes de refinar (remove quebras internas).",
    )
    r.add_argument(
        "--use-glossary",
        action="store_true",
        help="Ativa modo de glossario (manual + dinamico) nas chamadas de refine.",
    )
    r.add_argument(
        "--auto-glossary-dir",
        type=str,
        help="Diretorio opcional contendo varios JSONs de glossario manual (todos serao carregados).",
    )
    r.add_argument(
        "--manual-glossary",
        type=str,
        help="Arquivo JSON de glossario manual (somente leitura).",
    )
    r.add_argument(
        "--dynamic-glossary",
        type=str,
        help="Arquivo JSON de glossario dinamico (padrao: saida/glossario_dinamico.json).",
    )
    r.add_argument(
        "--debug-refine",
        action="store_true",
        help="Salva arquivos de debug (orig/raw/final) dos primeiros chunks de refine.",
    )
    r.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Numero de workers paralelos para refine (ordem preservada na montagem).",
    )
    r.add_argument(
        "--preprocess-advanced",
        action="store_true",
        help="Ativa pre-processamento avancado opcional no Markdown antes do refine.",
    )
    r.add_argument(
        "--cleanup-before-refine",
        choices=["off", "auto", "on"],
        default=None,
        help="Controle de limpeza deterministica antes do refine: off, auto, on.",
    )
    r.add_argument(
        "--debug-chunks",
        action="store_true",
        help="Ativa debug detalhado por chunk (JSONL) no refine.",
    )
    r.add_argument("--editor-lite", action="store_true", help="Ativa modo editor lite pos-refine.")
    r.add_argument("--editor-consistency", action="store_true", help="Ativa modo editor consistency pos-refine.")
    r.add_argument("--editor-voice", action="store_true", help="Ativa modo editor voice pos-refine.")
    r.add_argument("--editor-strict", action="store_true", help="Ativa modo editor strict pos-refine.")
    r.add_argument("--editor-report", action="store_true", help="Gera relatorio das mudancas do editor.")

    # Subcomando: PDF direto
    p = sub.add_parser(
        "pdf",
        parents=[common],
        help="Gera PDF a partir de um arquivo .md (ex.: *_pt_refinado.md).",
    )
    p.add_argument("--input", type=str, required=True, help="Arquivo .md para converter em PDF.")

    return parser


def find_pdfs(data_dir: Path, specific: str | None = None) -> list[Path]:
    """Lista PDFs no diretório ou retorna apenas o arquivo indicado."""
    if specific:
        return [Path(specific)]
    return sorted(p for p in data_dir.glob("*.pdf"))


def find_markdowns(output_dir: Path, specific: str | None = None) -> list[Path]:
    """Lista *_pt.md no diretório ou retorna apenas o arquivo indicado."""
    if specific:
        return [Path(specific)]
    return sorted(p for p in output_dir.glob("*_pt.md"))


def run_translate(args, cfg: AppConfig, logger: logging.Logger) -> None:
    """Executa pipeline completo de tradução (com refine opcional)."""
    ensure_paths(cfg)
    set_cache_base_dir(cfg.output_dir)
    if getattr(args, "clear_cache", None):
        clear_cache(args.clear_cache)
        logger.info("Cache %s limpo em %s", args.clear_cache, cfg.output_dir)
    if hasattr(args, "desquebrar_mode"):
        desquebrar_mode = getattr(args, "desquebrar_mode", getattr(cfg, "desquebrar_mode", "llm"))
    else:
        desquebrar_mode = "llm"
    pdfs = find_pdfs(cfg.data_dir, args.input)
    if not pdfs:
        raise SystemExit("Nenhum PDF encontrado em data/ ou caminho inválido.")

    use_desquebrar = bool(getattr(args, "use_desquebrar", getattr(cfg, "use_desquebrar", True)))

    backend = LLMBackend(
        backend=args.backend,
        model=args.model,
        temperature=cfg.translate_temperature,
        logger=logger,
        request_timeout=args.request_timeout,
        repeat_penalty=cfg.translate_repeat_penalty,
        num_predict=args.num_predict,
        num_ctx=cfg.translate_num_ctx,
        keep_alive=getattr(cfg, "ollama_keep_alive", "30m"),
    )
    logger.info(
        "LLM de tradução: backend=%s model=%s temp=%.2f chunk=%d timeout=%ds num_predict=%d",
        args.backend,
        args.model,
        cfg.translate_temperature,
        cfg.translate_chunk_chars,
        args.request_timeout,
        args.num_predict,
    )

    glossary_text = None
    glossary_state = None
    fail_on_chunk_error = getattr(args, "fail_on_chunk_error", None)
    if fail_on_chunk_error is None:
        fail_on_chunk_error = getattr(cfg, "fail_on_chunk_error", False)
    if getattr(args, "use_glossary", False):
        manual_path = Path(args.manual_glossary) if args.manual_glossary else Path("glossario/glossario_manual.json")
        glossary_state = build_glossary_state(manual_path=manual_path, dynamic_path=None, logger=logger, manual_dir=None)
        if glossary_state:
            glossary_text = format_manual_pairs_for_translation(glossary_state.manual_terms, limit=30)
            logger.info("Glossário manual carregado para tradução: %d termos (usando até 30 no prompt).", len(glossary_state.manual_terms))
        else:
            logger.warning("Uso de glossário solicitado, mas nenhum glossário manual carregado.")

    for pdf in pdfs:
        debug_run = None
        timings: dict[str, float] = {}
        current_stage = "init"
        output_refined: Path | None = None
        if args.debug:
            debug_run = DebugRunWriter.create(
                output_dir=cfg.output_dir,
                slug=pdf.stem,
                input_kind="pdf",
                max_chunks=cfg.debug_max_chunks,
                max_chars_per_file=cfg.debug_max_chars_per_file,
                store_llm_raw=cfg.debug_store_llm_raw,
            )
            translate_prompt_hash = translation_prompt_fingerprint(
                allow_adaptation=getattr(args, "translate_allow_adaptation", cfg.translate_allow_adaptation)
            )
            debug_run.write_run_metadata(
                args=vars(args),
                cfg=cfg,
                translate_prompt_hash=translate_prompt_hash,
                refine_prompt_hash=refine_prompt_fingerprint(),
            )
            backend_payload = {
                "translate": {
                    "backend": args.backend,
                    "model": args.model,
                    "temperature": cfg.translate_temperature,
                    "num_predict": args.num_predict,
                    "repeat_penalty": cfg.translate_repeat_penalty,
                },
                "refine": None,
                "desquebrar": {
                    "backend": args.desquebrar_backend,
                    "model": args.desquebrar_model,
                    "temperature": args.desquebrar_temperature,
                    "num_predict": args.desquebrar_num_predict,
                    "repeat_penalty": args.desquebrar_repeat_penalty,
                },
            }
            debug_run.write_backend(backend_payload)
        try:
            logger.info("Traduzindo PDF: %s", pdf.name)
            current_stage = "extract_pdf"
            start_stage = time.perf_counter()
            raw_text = extract_pdf_text(pdf, logger)
            timings["extract_pdf"] = time.perf_counter() - start_stage
            if not raw_text.strip():
                raise SystemExit(f"PDF {pdf.name} não possui texto extraído (pode ser imagem/scan).")
            if getattr(args, "preprocess_advanced", False):
                raw_text = advanced_clean(raw_text)
            if args.debug:
                logger.debug("Debug ativado: salvando também raw_extracted e preprocessed.")
                raw_out = cfg.output_dir / f"{pdf.stem}_raw_extracted.md"
                write_text(raw_out, raw_text)
                logger.info("Texto bruto salvo em %s", raw_out)
            if debug_run:
                debug_run.write_text(f"10_preprocess/{pdf.stem}_raw_extracted.md", raw_text)

            current_stage = "preprocess"
            start_stage = time.perf_counter()
            pre_text, pre_stats = preprocess_text(
                raw_text,
                logger,
                skip_front_matter=getattr(args, "skip_front_matter", cfg.skip_front_matter),
                return_stats=True,
            )
            timings["preprocess"] = time.perf_counter() - start_stage
            if args.debug:
                pre_out = cfg.output_dir / f"{pdf.stem}_preprocessed.md"
                write_text(pre_out, pre_text)
                logger.info("Texto preprocessado salvo em %s", pre_out)
            if debug_run:
                debug_run.preprocessed_rel = f"10_preprocess/{pdf.stem}_preprocessed.md"
                debug_run.write_text(debug_run.preprocessed_rel, pre_text)
                debug_run.write_preprocess_report(
                    {
                        "chars_in": pre_stats.get("chars_in", len(raw_text)),
                        "chars_out": pre_stats.get("chars_out", len(pre_text)),
                        "removed_chars": max(pre_stats.get("chars_in", len(raw_text)) - pre_stats.get("chars_out", len(pre_text)), 0),
                        "skip_front_matter": getattr(args, "skip_front_matter", cfg.skip_front_matter),
                        "oceanofpdf_removed_count": pre_stats.get("oceanofpdf_removed_count"),
                        "promo_lines_removed_count": pre_stats.get("promo_lines_removed_count"),
                        "urls_removed_count": pre_stats.get("urls_removed_count"),
                        "toc_blocks_removed_count": pre_stats.get("toc_blocks_removed_count"),
                        "remaining_counts": pre_stats.get("remaining_counts"),
                        "repeated_lines_removed_count": pre_stats.get("repeated_lines_removed_count"),
                        "top_repeated_lines": pre_stats.get("top_repeated_lines"),
                    }
                )

            working_text = pre_text
            desquebrar_stats = None
            if use_desquebrar:
                if desquebrar_mode == "safe":
                    current_stage = "desquebrar_safe"
                    logger.info("Modo safe: aplicando desquebrar_safe (sem LLM), preservando layout.")
                    working_text = desquebrar_safe(working_text)
                    if args.debug:
                        desq_out = cfg.output_dir / f"{pdf.stem}_raw_desquebrado.md"
                        write_text(desq_out, working_text)
                        logger.info("Texto desquebrado (safe) salvo em %s", desq_out)
                    if debug_run:
                        debug_run.desquebrado_rel = f"20_desquebrar/{pdf.stem}_raw_desquebrado.md"
                        debug_run.write_text(debug_run.desquebrado_rel, working_text)
                        debug_run.write_desquebrar_report(
                            {
                                "mode": "safe",
                                "chars_in": len(pre_text),
                                "chars_out": len(working_text),
                            }
                        )
                else:
                    current_stage = "desquebrar_llm"
                    logger.info(
                        "Aplicando desquebrar antes da tradução (backend=%s model=%s temp=%.2f chunk=%d num_predict=%d repeat_penalty=%s)",
                        args.desquebrar_backend,
                        args.desquebrar_model,
                        args.desquebrar_temperature,
                        args.desquebrar_chunk_chars,
                        args.desquebrar_num_predict,
                        args.desquebrar_repeat_penalty,
                    )
                    desquebrar_backend = LLMBackend(
                        backend=args.desquebrar_backend,
                        model=args.desquebrar_model,
                        temperature=args.desquebrar_temperature,
                        logger=logger,
                        request_timeout=args.request_timeout,
                        num_predict=args.desquebrar_num_predict,
                        repeat_penalty=args.desquebrar_repeat_penalty,
                        num_ctx=getattr(cfg, "desquebrar_num_ctx", None),
                        keep_alive=getattr(cfg, "ollama_keep_alive", "30m"),
                    )
                    start_stage = time.perf_counter()
                    working_text, desquebrar_stats = desquebrar_text(
                        working_text,
                        cfg,
                        logger,
                        backend=desquebrar_backend,
                        chunk_chars=args.desquebrar_chunk_chars,
                    )
                    timings["desquebrar"] = time.perf_counter() - start_stage
                    if desquebrar_stats:
                        logger.info(
                            "Desquebrar concluído: chunks=%d cache_hits=%d fallbacks=%d",
                            desquebrar_stats.total_chunks,
                            desquebrar_stats.cache_hits,
                            desquebrar_stats.fallbacks,
                        )
                    if args.debug:
                        desq_out = cfg.output_dir / f"{pdf.stem}_raw_desquebrado.md"
                        write_text(desq_out, working_text)
                        logger.info("Texto desquebrado salvo em %s", desq_out)
                    if debug_run:
                        debug_run.desquebrado_rel = f"20_desquebrar/{pdf.stem}_raw_desquebrado.md"
                        debug_run.write_text(debug_run.desquebrado_rel, working_text)
                        debug_run.write_desquebrar_report(
                            {
                                "mode": "llm",
                                "stats": desquebrar_stats_to_dict(desquebrar_stats, cfg) if desquebrar_stats else {},
                            }
                        )
                    try:
                        metrics_path = cfg.output_dir / f"{pdf.stem}_desquebrar_metrics.json"
                        metrics_payload = desquebrar_stats_to_dict(desquebrar_stats, cfg)
                        metrics_payload["timestamp"] = datetime.now().isoformat()
                        metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception as exc:
                        logger.warning("Falha ao gravar métricas do desquebrar: %s", exc)
            else:
                current_stage = "desquebrar_off"
                logger.info("Desquebrar desativado; seguindo direto para tradução.")
                if debug_run:
                    debug_run.desquebrado_rel = f"20_desquebrar/{pdf.stem}_raw_desquebrado.md"
                    debug_run.write_text(debug_run.desquebrado_rel, working_text)
                    debug_run.write_desquebrar_report(
                        {
                            "mode": "off",
                            "chars_in": len(pre_text),
                            "chars_out": len(working_text),
                        }
                    )

            progress_path = cfg.output_dir / f"{pdf.stem}_pt_progress.json"
            resume_manifest = None
            if args.resume:
                try:
                    loaded = json.loads(progress_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        resume_manifest = loaded
                    else:
                        logger.warning(
                            "Manifesto de progresso %s tem formato inesperado; traduzindo do zero.",
                            progress_path,
                        )
                except FileNotFoundError:
                    logger.warning(
                        "Manifesto de progresso não encontrado em %s; tradução completa será executada.",
                        progress_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "Falha ao ler manifesto de progresso %s (%s); tradução completa será executada.",
                        progress_path,
                        exc,
                    )

            current_stage = "translate"
            start_stage = time.perf_counter()
            translated_md = translate_document(
                pdf_text=working_text,
                backend=backend,
                cfg=cfg,
                logger=logger,
                source_slug=pdf.stem,
                progress_path=progress_path,
                resume_manifest=resume_manifest,
                glossary_text=glossary_text,
                glossary_manual_terms=glossary_state.manual_terms if glossary_state else None,
                debug_translation=getattr(args, "debug", False),
                parallel_workers=max(1, getattr(args, "parallel", 1)),
                debug_chunks=getattr(args, "debug_chunks", False),
                already_preprocessed=True,
                split_by_sections=getattr(args, "split_by_sections", cfg.split_by_sections),
                allow_adaptation=getattr(args, "translate_allow_adaptation", cfg.translate_allow_adaptation),
                fail_on_chunk_error=fail_on_chunk_error,
                debug_run=debug_run,
            )
            timings["translate"] = time.perf_counter() - start_stage

            md_path = cfg.output_dir / f"{pdf.stem}_pt.md"
            write_text(md_path, translated_md)
            logger.info("Markdown salvo em %s", md_path)
            if debug_run:
                debug_run.pt_output_rel = str(md_path.relative_to(cfg.output_dir))

            logger.info("Conversão para PDF desativada temporariamente; saída principal é o arquivo .md.")

            if args.no_refine:
                logger.info("Refinamento desabilitado (--no-refine); apenas *_pt.md será gerado.")
            else:
                current_stage = "refine"
                logger.info("Executando refine opcional para %s", md_path.name)
                cleanup_mode = args.cleanup_before_refine or getattr(cfg, "cleanup_before_refine", "off")
                if cleanup_mode not in ("off", "auto", "on"):
                    cleanup_mode = "off"
                refine_backend = LLMBackend(
                    backend=cfg.refine_backend,
                    model=cfg.refine_model,
                    temperature=cfg.refine_temperature,
                    logger=logger,
                    request_timeout=args.request_timeout,
                    repeat_penalty=cfg.refine_repeat_penalty,
                    num_predict=cfg.refine_num_predict,
                    num_ctx=getattr(cfg, "refine_num_ctx", None),
                    keep_alive=getattr(cfg, "ollama_keep_alive", "30m"),
                )
                logger.info(
                    "LLM de refine (opcional): backend=%s model=%s temp=%.2f chunk=%d timeout=%ds num_predict=%d",
                    cfg.refine_backend,
                    cfg.refine_model,
                    cfg.refine_temperature,
                    cfg.refine_chunk_chars,
                    args.request_timeout,
                    cfg.refine_num_predict,
                )
                if debug_run:
                    backend_payload = {
                        "translate": {
                            "backend": args.backend,
                            "model": args.model,
                            "temperature": cfg.translate_temperature,
                            "num_predict": args.num_predict,
                            "repeat_penalty": cfg.translate_repeat_penalty,
                        },
                        "refine": {
                            "backend": cfg.refine_backend,
                            "model": cfg.refine_model,
                            "temperature": cfg.refine_temperature,
                            "num_predict": cfg.refine_num_predict,
                            "repeat_penalty": cfg.refine_repeat_penalty,
                        },
                        "desquebrar": {
                            "backend": args.desquebrar_backend,
                            "model": args.desquebrar_model,
                            "temperature": args.desquebrar_temperature,
                            "num_predict": args.desquebrar_num_predict,
                            "repeat_penalty": args.desquebrar_repeat_penalty,
                        },
                    }
                    debug_run.write_backend(backend_payload)
                output_refined = cfg.output_dir / f"{pdf.stem}_pt_refinado.md"
                start_stage = time.perf_counter()
                refine_markdown_file(
                    input_path=md_path,
                    output_path=output_refined,
                    backend=refine_backend,
                    cfg=cfg,
                    logger=logger,
                    progress_path=cfg.output_dir / f"{pdf.stem}_pt_refinado_progress.json",
                    resume_manifest=None,
                    debug_chunks=getattr(args, "debug_chunks", False),
                    cleanup_mode=cleanup_mode,
                    debug_run=debug_run,
                )
                timings["refine"] = time.perf_counter() - start_stage
                logger.info("Conversão para PDF desativada temporariamente; saída principal é o arquivo .md refinado.")
                try:
                    refined_text = read_text(output_refined)
                    refined_text = normalize_structure(refined_text)
                    write_text(output_refined, refined_text)
                except Exception as exc:
                    logger.warning("Falha ao normalizar estrutura do refinado: %s", exc)
                if debug_run:
                    debug_run.pt_refined_rel = str(output_refined.relative_to(cfg.output_dir))
                pdf_enabled = bool(getattr(args, "pdf_enabled", cfg.pdf_enabled))
                if pdf_enabled:
                    try:
                        pdf_dir = cfg.output_dir / "pdf"
                        pdf_output = pdf_dir / f"{output_refined.stem}.pdf"
                        convert_markdown_to_pdf(
                            md_path=output_refined,
                            output_path=pdf_output,
                            cfg=cfg,
                            logger=logger,
                            title=output_refined.stem,
                        )
                        logger.info("PDF gerado em %s", pdf_output)
                    except Exception as exc:
                        logger.error("Falha ao gerar PDF automaticamente: %s", exc)
            if debug_run:
                run_dir_rel = str(debug_run.run_dir.relative_to(cfg.output_dir))
                summary = {
                    "run_id": debug_run.run_id,
                    "source_slug": pdf.stem,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input_kind": debug_run.input_kind,
                    "paths": {
                        "run_dir": run_dir_rel,
                        "preprocessed": debug_run.preprocessed_rel,
                        "desquebrado": debug_run.desquebrado_rel,
                        "chunks": "30_split_chunk/chunks.jsonl",
                        "sections": "30_split_chunk/sections.json",
                        "translate_manifest": "40_translate/translate_manifest.json",
                        "refine_manifest": "60_refine/refine_manifest.json",
                        "timings": "99_reports/timings.json",
                        "errors": "99_reports/errors.jsonl",
                    },
                    "final_outputs": {
                        "pt": debug_run.pt_output_rel,
                        "pt_refinado": debug_run.pt_refined_rel,
                    },
                    "hashes": {
                        "preprocessed": debug_run.sha256_text(pre_text),
                        "desquebrado": debug_run.sha256_text(working_text),
                        "pt": debug_run.sha256_text(translated_md),
                        "pt_refinado": debug_run.sha256_text(read_text(output_refined)) if output_refined else None,
                    },
                    "notes": [],
                }
                debug_run.write_run_summary(summary)
                timings["total"] = sum(timings.values())
                debug_run.write_timing(timings)
        except Exception as exc:
            if debug_run:
                debug_run.write_error(
                    {
                        "stage": current_stage,
                        "message": str(exc),
                        "stack": traceback.format_exc(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            raise


def run_translate_md(args, cfg: AppConfig, logger: logging.Logger) -> None:
    """Traduz um arquivo .md/.txt j  desquebrado diretamente, com refine opcional."""
    ensure_paths(cfg)
    set_cache_base_dir(cfg.output_dir)
    if getattr(args, "clear_cache", None):
        clear_cache(args.clear_cache)
        logger.info("Cache %s limpo em %s", args.clear_cache, cfg.output_dir)

    text_path = Path(args.input)
    if not text_path.exists():
        raise SystemExit(f"Arquivo nÆo encontrado: {text_path}")

    debug_run = None
    timings: dict[str, float] = {}
    current_stage = "init"
    output_refined: Path | None = None
    if args.debug:
        debug_run = DebugRunWriter.create(
            output_dir=cfg.output_dir,
            slug=text_path.stem,
            input_kind="md",
            max_chunks=cfg.debug_max_chunks,
            max_chars_per_file=cfg.debug_max_chars_per_file,
            store_llm_raw=cfg.debug_store_llm_raw,
        )
        translate_prompt_hash = translation_prompt_fingerprint(
            allow_adaptation=getattr(args, "translate_allow_adaptation", cfg.translate_allow_adaptation)
        )
        debug_run.write_run_metadata(
            args=vars(args),
            cfg=cfg,
            translate_prompt_hash=translate_prompt_hash,
            refine_prompt_hash=refine_prompt_fingerprint(),
        )
        backend_payload = {
            "translate": {
                "backend": args.backend,
                "model": args.model,
                "temperature": cfg.translate_temperature,
                "num_predict": args.num_predict,
                "repeat_penalty": cfg.translate_repeat_penalty,
            },
            "refine": None,
            "desquebrar": None,
        }
        debug_run.write_backend(backend_payload)

    try:
        current_stage = "load_input"
        start_stage = time.perf_counter()
        raw_text = read_text(text_path)
        timings["load_input"] = time.perf_counter() - start_stage
        if not raw_text.strip():
            raise SystemExit(f"Arquivo {text_path} vazio.")

        current_stage = "preprocess"
        if getattr(args, "preprocess_advanced", False):
            raw_text = advanced_clean(raw_text)
        if getattr(args, "normalize_paragraphs", False):
            raw_text = normalize_md_paragraphs(raw_text)
        if debug_run:
            debug_run.preprocessed_rel = f"10_preprocess/{text_path.stem}_preprocessed.md"
            debug_run.write_text(debug_run.preprocessed_rel, raw_text)
            debug_run.write_preprocess_report(
                {
                    "chars_in": len(raw_text),
                    "chars_out": len(raw_text),
                    "removed_chars": 0,
                    "skip_front_matter": False,
                }
            )
            debug_run.desquebrado_rel = f"20_desquebrar/{text_path.stem}_raw_desquebrado.md"
            debug_run.write_text(debug_run.desquebrado_rel, raw_text)
            debug_run.write_desquebrar_report(
                {
                    "mode": "input",
                    "chars_in": len(raw_text),
                    "chars_out": len(raw_text),
                }
            )

        backend = LLMBackend(
            backend=args.backend,
            model=args.model,
            temperature=cfg.translate_temperature,
            logger=logger,
            request_timeout=args.request_timeout,
            repeat_penalty=cfg.translate_repeat_penalty,
            num_predict=args.num_predict,
            num_ctx=cfg.translate_num_ctx,
            keep_alive=getattr(cfg, "ollama_keep_alive", "30m"),
        )
        logger.info(
            "LLM de tradu‡Æo: backend=%s model=%s temp=%.2f chunk=%d timeout=%ds num_predict=%d",
            args.backend,
            args.model,
            cfg.translate_temperature,
            cfg.translate_chunk_chars,
            args.request_timeout,
            args.num_predict,
        )

        glossary_text = None
        glossary_state = None
        fail_on_chunk_error = getattr(args, "fail_on_chunk_error", None)
        if fail_on_chunk_error is None:
            fail_on_chunk_error = getattr(cfg, "fail_on_chunk_error", False)
        if getattr(args, "use_glossary", False):
            manual_path = Path(args.manual_glossary) if args.manual_glossary else Path("glossario/glossario_manual.json")
            glossary_state = build_glossary_state(manual_path=manual_path, dynamic_path=None, logger=logger, manual_dir=None)
            if glossary_state:
                glossary_text = format_manual_pairs_for_translation(glossary_state.manual_terms, limit=30)
                logger.info("Gloss rio manual carregado para tradu‡Æo: %d termos (usando at‚ 30 no prompt).", len(glossary_state.manual_terms))
            else:
                logger.warning("Uso de gloss rio solicitado, mas nenhum gloss rio manual carregado.")

        progress_path = cfg.output_dir / f"{text_path.stem}_pt_progress.json"
        resume_manifest = None
        if getattr(args, "resume", False):
            try:
                loaded = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    resume_manifest = loaded
                else:
                    logger.warning(
                        "Manifesto de progresso %s tem formato inesperado; traduzindo do zero.",
                        progress_path,
                    )
            except FileNotFoundError:
                logger.warning(
                    "Manifesto de progresso nÆo encontrado em %s; tradu‡Æo completa ser  executada.",
                    progress_path,
                )
            except Exception as exc:
                logger.warning(
                    "Falha ao ler manifesto de progresso %s (%s); tradu‡Æo completa ser  executada.",
                    progress_path,
                    exc,
                )

        current_stage = "translate"
        start_stage = time.perf_counter()
        translated_md = translate_document(
            pdf_text=raw_text,
            backend=backend,
            cfg=cfg,
            logger=logger,
            source_slug=text_path.stem,
            progress_path=progress_path,
            resume_manifest=resume_manifest,
            glossary_text=glossary_text,
            glossary_manual_terms=glossary_state.manual_terms if glossary_state else None,
            debug_translation=getattr(args, "debug", False),
            parallel_workers=max(1, getattr(args, "parallel", 1)),
            debug_chunks=getattr(args, "debug_chunks", False),
            already_preprocessed=True,
            split_by_sections=getattr(args, "split_by_sections", cfg.split_by_sections),
            allow_adaptation=getattr(args, "translate_allow_adaptation", cfg.translate_allow_adaptation),
            fail_on_chunk_error=fail_on_chunk_error,
            debug_run=debug_run,
        )
        timings["translate"] = time.perf_counter() - start_stage

        md_path = cfg.output_dir / f"{text_path.stem}_pt.md"
        write_text(md_path, translated_md)
        logger.info("Markdown salvo em %s", md_path)
        if debug_run:
            debug_run.pt_output_rel = str(md_path.relative_to(cfg.output_dir))

        if args.no_refine:
            logger.info("Refinamento desabilitado (--no-refine); apenas *_pt.md ser  gerado.")
            if debug_run:
                run_dir_rel = str(debug_run.run_dir.relative_to(cfg.output_dir))
                summary = {
                    "run_id": debug_run.run_id,
                    "source_slug": text_path.stem,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input_kind": debug_run.input_kind,
                    "paths": {
                        "run_dir": run_dir_rel,
                        "preprocessed": debug_run.preprocessed_rel,
                        "desquebrado": debug_run.desquebrado_rel,
                        "chunks": "30_split_chunk/chunks.jsonl",
                        "sections": "30_split_chunk/sections.json",
                        "translate_manifest": "40_translate/translate_manifest.json",
                        "refine_manifest": "60_refine/refine_manifest.json",
                        "timings": "99_reports/timings.json",
                        "errors": "99_reports/errors.jsonl",
                    },
                    "final_outputs": {
                        "pt": debug_run.pt_output_rel,
                        "pt_refinado": None,
                    },
                    "hashes": {
                        "preprocessed": debug_run.sha256_text(raw_text),
                        "desquebrado": debug_run.sha256_text(raw_text),
                        "pt": debug_run.sha256_text(translated_md),
                        "pt_refinado": None,
                    },
                    "notes": [],
                }
                debug_run.write_run_summary(summary)
                timings["total"] = sum(timings.values())
                debug_run.write_timing(timings)
            return

        current_stage = "refine"
        logger.info("Executando refine opcional para %s", md_path.name)
        cleanup_mode = args.cleanup_before_refine or getattr(cfg, "cleanup_before_refine", "off")
        if cleanup_mode not in ("off", "auto", "on"):
            cleanup_mode = "off"
        refine_backend = LLMBackend(
            backend=cfg.refine_backend,
            model=cfg.refine_model,
            temperature=cfg.refine_temperature,
            logger=logger,
            request_timeout=args.request_timeout,
            repeat_penalty=cfg.refine_repeat_penalty,
            num_predict=cfg.refine_num_predict,
            num_ctx=getattr(cfg, "refine_num_ctx", None),
            keep_alive=getattr(cfg, "ollama_keep_alive", "30m"),
        )
        logger.info(
            "LLM de refine (opcional): backend=%s model=%s temp=%.2f chunk=%d timeout=%ds num_predict=%d",
            cfg.refine_backend,
            cfg.refine_model,
            cfg.refine_temperature,
            cfg.refine_chunk_chars,
            args.request_timeout,
            cfg.refine_num_predict,
        )
        if debug_run:
            backend_payload = {
                "translate": {
                    "backend": args.backend,
                    "model": args.model,
                    "temperature": cfg.translate_temperature,
                    "num_predict": args.num_predict,
                    "repeat_penalty": cfg.translate_repeat_penalty,
                },
                "refine": {
                    "backend": cfg.refine_backend,
                    "model": cfg.refine_model,
                    "temperature": cfg.refine_temperature,
                    "num_predict": cfg.refine_num_predict,
                    "repeat_penalty": cfg.refine_repeat_penalty,
                },
                "desquebrar": None,
            }
            debug_run.write_backend(backend_payload)
        output_refined = cfg.output_dir / f"{text_path.stem}_pt_refinado.md"
        start_stage = time.perf_counter()
        refine_markdown_file(
            input_path=md_path,
            output_path=output_refined,
            backend=refine_backend,
            cfg=cfg,
            logger=logger,
            progress_path=cfg.output_dir / f"{text_path.stem}_pt_refinado_progress.json",
            resume_manifest=None,
            debug_chunks=getattr(args, "debug_chunks", False),
            cleanup_mode=cleanup_mode,
            debug_run=debug_run,
        )
        timings["refine"] = time.perf_counter() - start_stage
        logger.info("ConversÆo para PDF desativada temporariamente; sa¡da principal ‚ o arquivo .md refinado.")
        try:
            refined_text = read_text(output_refined)
            refined_text = normalize_structure(refined_text)
            write_text(output_refined, refined_text)
        except Exception as exc:
            logger.warning("Falha ao normalizar estrutura do refinado: %s", exc)
        pdf_enabled = bool(getattr(args, "pdf_enabled", cfg.pdf_enabled))
        if pdf_enabled:
            try:
                pdf_dir = cfg.output_dir / "pdf"
                pdf_output = pdf_dir / f"{output_refined.stem}.pdf"
                convert_markdown_to_pdf(
                    md_path=output_refined,
                    output_path=pdf_output,
                    cfg=cfg,
                    logger=logger,
                    title=output_refined.stem,
                )
                logger.info("PDF gerado em %s", pdf_output)
            except Exception as exc:
                logger.error("Falha ao gerar PDF automaticamente: %s", exc)
        if debug_run:
            debug_run.pt_refined_rel = str(output_refined.relative_to(cfg.output_dir))
            run_dir_rel = str(debug_run.run_dir.relative_to(cfg.output_dir))
            summary = {
                "run_id": debug_run.run_id,
                "source_slug": text_path.stem,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_kind": debug_run.input_kind,
                "paths": {
                    "run_dir": run_dir_rel,
                    "preprocessed": debug_run.preprocessed_rel,
                    "desquebrado": debug_run.desquebrado_rel,
                    "chunks": "30_split_chunk/chunks.jsonl",
                    "sections": "30_split_chunk/sections.json",
                    "translate_manifest": "40_translate/translate_manifest.json",
                    "refine_manifest": "60_refine/refine_manifest.json",
                    "timings": "99_reports/timings.json",
                    "errors": "99_reports/errors.jsonl",
                },
                "final_outputs": {
                    "pt": debug_run.pt_output_rel,
                    "pt_refinado": debug_run.pt_refined_rel,
                },
                "hashes": {
                    "preprocessed": debug_run.sha256_text(raw_text),
                    "desquebrado": debug_run.sha256_text(raw_text),
                    "pt": debug_run.sha256_text(translated_md),
                    "pt_refinado": debug_run.sha256_text(read_text(output_refined)),
                },
                "notes": [],
            }
            debug_run.write_run_summary(summary)
            timings["total"] = sum(timings.values())
            debug_run.write_timing(timings)
    except Exception as exc:
        if debug_run:
            debug_run.write_error(
                {
                    "stage": current_stage,
                    "message": str(exc),
                    "stack": traceback.format_exc(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        raise


def run_refine(args, cfg: AppConfig, logger: logging.Logger) -> None:
    """Executa refine sobre arquivos *_pt.md existentes."""
    ensure_paths(cfg)
    set_cache_base_dir(cfg.output_dir)
    if getattr(args, "clear_cache", None):
        clear_cache(args.clear_cache)
        logger.info("Cache %s limpo em %s", args.clear_cache, cfg.output_dir)
    if hasattr(args, "desquebrar_mode"):
        desquebrar_mode = getattr(args, "desquebrar_mode", getattr(cfg, "desquebrar_mode", "llm"))
    else:
        desquebrar_mode = "llm"
    if desquebrar_mode == "safe":
        logger.info("Modo safe afeta apenas o desquebrar; comando refina usa fluxo padrão de refine.")
    md_files = find_markdowns(cfg.output_dir, args.input)
    if not md_files:
        raise SystemExit("Nenhum *_pt.md encontrado em saida/ ou caminho inválido.")

    backend = LLMBackend(
        backend=args.backend,
        model=args.model,
        temperature=cfg.refine_temperature,
        logger=logger,
        request_timeout=args.request_timeout,
        repeat_penalty=cfg.refine_repeat_penalty,
        num_predict=args.num_predict,
        num_ctx=getattr(cfg, "refine_num_ctx", None),
        keep_alive=getattr(cfg, "ollama_keep_alive", "30m"),
    )
    logger.info(
        "LLM de refine: backend=%s model=%s temp=%.2f chunk=%d timeout=%ds num_predict=%d",
        args.backend,
        args.model,
        cfg.refine_temperature,
        cfg.refine_chunk_chars,
        args.request_timeout,
        args.num_predict,
    )

    glossary_state = None
    cleanup_mode = args.cleanup_before_refine or getattr(cfg, "cleanup_before_refine", "off")
    if cleanup_mode not in ("off", "auto", "on"):
        cleanup_mode = "off"
    if getattr(args, "use_glossary", False):
        manual_path = Path(args.manual_glossary) if args.manual_glossary else None
        manual_dir = Path(args.auto_glossary_dir) if getattr(args, "auto_glossary_dir", None) else None
        dynamic_path = Path(args.dynamic_glossary) if args.dynamic_glossary else cfg.output_dir / "glossario_dinamico.json"
        logger.info(
            "Modo glossário ativo. Manual: %s | Dinâmico: %s | Auto-dir: %s",
            manual_path if manual_path else "nenhum",
            dynamic_path,
            manual_dir if manual_dir else "nenhum",
        )
        glossary_state = build_glossary_state(manual_path, dynamic_path, logger, manual_dir=manual_dir)

    for md in md_files:
        stem = md.stem.replace("_pt", "")
        output_md = cfg.output_dir / f"{stem}_pt_refinado.md"
        output_pdf = cfg.output_dir / f"{stem}_pt_refinado.pdf"
        progress_path = cfg.output_dir / f"{stem}_pt_refinado_progress.json"
        resume_manifest = None
        if getattr(args, "resume", False):
            try:
                loaded = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    resume_manifest = loaded
                else:
                    logger.warning(
                        "Manifesto de progresso %s tem formato inesperado; refinando do zero.",
                        progress_path,
                    )
            except FileNotFoundError:
                logger.warning(
                    "Manifesto de progresso não encontrado em %s; refine completo será executado.",
                    progress_path,
                )
            except Exception as exc:
                logger.warning(
                    "Falha ao ler manifesto de progresso %s (%s); refine completo será executado.",
                    progress_path,
                    exc,
                )
        refine_markdown_file(
            input_path=md,
            output_path=output_md,
            backend=backend,
            cfg=cfg,
            logger=logger,
            progress_path=progress_path,
            resume_manifest=resume_manifest,
            normalize_paragraphs=getattr(args, "normalize_paragraphs", False),
            glossary_state=glossary_state,
            debug_refine=getattr(args, "debug_refine", False),
            parallel_workers=max(1, getattr(args, "parallel", 1)),
            preprocess_advanced=getattr(args, "preprocess_advanced", False),
            debug_chunks=getattr(args, "debug_chunks", False),
            cleanup_mode=cleanup_mode,
        )
        # pós-processamento final em PT-BR antes de PDF
        refined_text = read_text(output_md)
        editor_flags = {
            "lite": getattr(args, "editor_lite", False),
            "consistency": getattr(args, "editor_consistency", False),
            "voice": getattr(args, "editor_voice", False),
            "strict": getattr(args, "editor_strict", False),
        }
        editor_changes = []
        if any(editor_flags.values()):
            refined_text, editor_changes = editor_pipeline(refined_text, editor_flags)
            if getattr(args, "editor_report", False):
                report_path = cfg.output_dir / "editor_report.json"
                report_payload = {
                    "modes": [k for k, v in editor_flags.items() if v],
                    "changes": editor_changes,
                }
                report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        refined_text = final_pt_postprocess(refined_text)
        refined_text = normalize_structure(refined_text)
        write_text(output_md, refined_text)
        markdown_to_pdf(
            markdown_text=refined_text,
            output_path=output_pdf,
            font_dir=cfg.font_dir,
            title_size=cfg.pdf_title_font_size,
            heading_size=cfg.pdf_heading_font_size,
            body_size=cfg.pdf_body_font_size,
            logger=logger,
        )


def run_pdf(args, cfg: AppConfig, logger: logging.Logger) -> None:
    """Gera PDF a partir de um arquivo .md existente."""
    ensure_paths(cfg)
    md_path = Path(args.input)
    if not md_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {md_path}")
    pdf_dir = cfg.output_dir / "pdf"
    pdf_output = pdf_dir / f"{md_path.stem}.pdf"
    convert_markdown_to_pdf(md_path=md_path, output_path=pdf_output, cfg=cfg, logger=logger, title=md_path.stem)
    logger.info("PDF gerado em %s", pdf_output)


def main() -> None:
    cfg = load_config()
    parser = build_parser(cfg)
    args = parser.parse_args()
    logger = setup_logging(logging.DEBUG if args.debug else logging.INFO)

    if args.command == "traduz":
        run_translate(args, cfg, logger)
    elif args.command == "traduz-md":
        run_translate_md(args, cfg, logger)
    elif args.command == "refina":
        run_refine(args, cfg, logger)
    elif args.command == "pdf":
        run_pdf(args, cfg, logger)
    else:
        parser.error("Comando inválido.")


if __name__ == "__main__":
    main()
