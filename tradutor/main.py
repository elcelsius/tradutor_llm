"""
CLI principal para tradução e refine.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from .config import AppConfig, ensure_paths, load_config
from .llm_backend import LLMBackend
from .pdf_export import markdown_to_pdf
from .pdf_reader import extract_pdf_text
from .preprocess import preprocess_text
from .refine import refine_markdown_file
from .translate import translate_document
from .utils import setup_logging, write_text, read_text


def build_parser(cfg: AppConfig) -> argparse.ArgumentParser:
    """Constroi o parser de argumentos com subcomandos traduz/refina."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--debug",
        action="store_true",
        help="Ativa logs detalhados e artefatos intermediários.",
    )  # permite --debug antes ou depois do subcomando
    parser = argparse.ArgumentParser(
        description="Tradutor e refinador de PDFs com LLMs.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcomando: traduzir
    t = sub.add_parser(
        "traduz",
        parents=[common],
        help="Traduz PDFs da pasta data/ (ou um arquivo específico).",
    )
    t.add_argument("--input", type=str, help="PDF específico para traduzir.")
    t.add_argument("--backend", type=str, choices=["ollama", "gemini"], default=cfg.translate_backend)
    t.add_argument("--model", type=str, default=cfg.translate_model)
    t.add_argument("--no-refine", action="store_true", help="Não executar refine após traduzir.")
    t.add_argument(
        "--resume",
        action="store_true",
        help="Retoma tradução usando manifesto de progresso existente (se houver).",
    )

    # Subcomando: refinar
    r = sub.add_parser(
        "refina",
        parents=[common],
        help="Refina arquivos *_pt.md na pasta saida/ ou um arquivo específico.",
    )
    r.add_argument("--input", type=str, help="Arquivo específico para refinar (ex.: saida/xxx_pt.md).")
    r.add_argument("--backend", type=str, choices=["ollama", "gemini"], default=cfg.refine_backend)
    r.add_argument("--model", type=str, default=cfg.refine_model)
    r.add_argument(
        "--resume",
        action="store_true",
        help="Retoma refine usando manifesto de progresso existente (se houver).",
    )

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
    pdfs = find_pdfs(cfg.data_dir, args.input)
    if not pdfs:
        raise SystemExit("Nenhum PDF encontrado em data/ ou caminho inválido.")

    backend = LLMBackend(
        backend=args.backend,
        model=args.model,
        temperature=cfg.translate_temperature,
        logger=logger,
        request_timeout=cfg.request_timeout,
    )
    logger.info(
        "LLM de tradução: backend=%s model=%s temp=%.2f chunk=%d",
        args.backend,
        args.model,
        cfg.translate_temperature,
        cfg.translate_chunk_chars,
    )

    for pdf in pdfs:
        logger.info("Traduzindo PDF: %s", pdf.name)
        raw_text = extract_pdf_text(pdf, logger)
        if not raw_text.strip():
            raise SystemExit(f"PDF {pdf.name} não possui texto extraído (pode ser imagem/scan).")
        if args.debug:
            logger.debug("Debug ativado: salvando também raw_extract e preprocessed.")
            raw_out = cfg.output_dir / f"{pdf.stem}_raw_extract.md"
            write_text(raw_out, raw_text)
            logger.info("Texto bruto salvo em %s", raw_out)

        pre_text = preprocess_text(raw_text, logger)
        if args.debug:
            pre_out = cfg.output_dir / f"{pdf.stem}_preprocessed.md"
            write_text(pre_out, pre_text)
            logger.info("Texto preprocessado salvo em %s", pre_out)

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

        translated_md = translate_document(
            pdf_text=pre_text,
            backend=backend,
            cfg=cfg,
            logger=logger,
            source_slug=pdf.stem,
            progress_path=progress_path,
            resume_manifest=resume_manifest,
        )

        md_path = cfg.output_dir / f"{pdf.stem}_pt.md"
        write_text(md_path, translated_md)
        logger.info("Markdown salvo em %s", md_path)

        logger.info("Conversão para PDF desativada temporariamente; saída principal é o arquivo .md.")

        if args.no_refine:
            logger.info("Refinamento desabilitado (--no-refine); apenas *_pt.md será gerado.")
        else:
            logger.info("Executando refine opcional para %s", md_path.name)
            refine_backend = LLMBackend(
                backend=cfg.refine_backend,
                model=cfg.refine_model,
                temperature=cfg.refine_temperature,
                logger=logger,
                request_timeout=cfg.request_timeout,
            )
            logger.info(
                "LLM de refine (opcional): backend=%s model=%s temp=%.2f chunk=%d",
                cfg.refine_backend,
                cfg.refine_model,
                cfg.refine_temperature,
                cfg.refine_chunk_chars,
            )
            output_refined = cfg.output_dir / f"{pdf.stem}_pt_refinado.md"
            refine_markdown_file(
                input_path=md_path,
                output_path=output_refined,
                backend=refine_backend,
                cfg=cfg,
                logger=logger,
                progress_path=cfg.output_dir / f"{pdf.stem}_pt_refinado_progress.json",
                resume_manifest=None,
            )
            logger.info("Conversão para PDF desativada temporariamente; saída principal é o arquivo .md refinado.")


def run_refine(args, cfg: AppConfig, logger: logging.Logger) -> None:
    """Executa refine sobre arquivos *_pt.md existentes."""
    ensure_paths(cfg)
    md_files = find_markdowns(cfg.output_dir, args.input)
    if not md_files:
        raise SystemExit("Nenhum *_pt.md encontrado em saida/ ou caminho inválido.")

    backend = LLMBackend(
        backend=args.backend,
        model=args.model,
        temperature=cfg.refine_temperature,
        logger=logger,
        request_timeout=cfg.request_timeout,
    )
    logger.info(
        "LLM de refine: backend=%s model=%s temp=%.2f chunk=%d",
        args.backend,
        args.model,
        cfg.refine_temperature,
        cfg.refine_chunk_chars,
    )

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
        )
        markdown_to_pdf(
            markdown_text=read_text(output_md),
            output_path=output_pdf,
            font_dir=cfg.font_dir,
            title_size=cfg.pdf_title_font_size,
            heading_size=cfg.pdf_heading_font_size,
            body_size=cfg.pdf_body_font_size,
            logger=logger,
        )


def main() -> None:
    cfg = load_config()
    parser = build_parser(cfg)
    args = parser.parse_args()
    logger = setup_logging(logging.DEBUG if args.debug else logging.INFO)

    if args.command == "traduz":
        run_translate(args, cfg, logger)
    elif args.command == "refina":
        run_refine(args, cfg, logger)
    else:
        parser.error("Comando inválido.")


if __name__ == "__main__":
    main()
