"""
CLI principal para tradução e refine.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from .config import AppConfig, ensure_paths
from .llm_backend import LLMBackend
from .pdf_export import markdown_to_pdf
from .preprocess import extract_text_from_pdf
from .refine import refine_markdown_file
from .translate import translate_document
from .utils import setup_logging, write_text, read_text


def build_parser(cfg: AppConfig) -> argparse.ArgumentParser:
    """Constroi o parser de argumentos com subcomandos traduz/refina."""
    parser = argparse.ArgumentParser(description="Tradutor e refinador de PDFs com LLMs.")
    parser.add_argument("--debug", action="store_true", help="Ativa logs de depuração.")
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcomando: traduzir
    t = sub.add_parser("traduz", help="Traduz PDFs da pasta data/ (ou um arquivo específico).")
    t.add_argument("--input", type=str, help="PDF específico para traduzir.")
    t.add_argument("--backend", type=str, choices=["ollama", "gemini"], default=cfg.translate_backend)
    t.add_argument("--model", type=str, default=cfg.translate_model)
    t.add_argument("--no-refine", action="store_true", help="Não executar refine após traduzir.")

    # Subcomando: refinar
    r = sub.add_parser("refina", help="Refina arquivos *_pt.md na pasta saida/ ou um arquivo específico.")
    r.add_argument("--input", type=str, help="Arquivo específico para refinar (ex.: saida/xxx_pt.md).")
    r.add_argument("--backend", type=str, choices=["ollama", "gemini"], default=cfg.refine_backend)
    r.add_argument("--model", type=str, default=cfg.refine_model)

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

    for pdf in pdfs:
        logger.info("Traduzindo PDF: %s", pdf.name)
        raw_text = extract_text_from_pdf(pdf, logger)
        translated_md = translate_document(pdf_text=raw_text, backend=backend, cfg=cfg, logger=logger)

        md_path = cfg.output_dir / f"{pdf.stem}_pt.md"
        pdf_path = cfg.output_dir / f"{pdf.stem}_pt.pdf"
        write_text(md_path, translated_md)
        logger.info("Markdown salvo em %s", md_path)

        markdown_to_pdf(
            markdown_text=translated_md,
            output_path=pdf_path,
            font_dir=cfg.font_dir,
            title_size=cfg.pdf_title_font_size,
            heading_size=cfg.pdf_heading_font_size,
            body_size=cfg.pdf_body_font_size,
            logger=logger,
        )

        if not args.no_refine:
            logger.info("Executando refine opcional para %s", md_path.name)
            refine_backend = LLMBackend(
                backend=cfg.refine_backend,
                model=cfg.refine_model,
                temperature=cfg.refine_temperature,
                logger=logger,
                request_timeout=cfg.request_timeout,
            )
            output_refined = cfg.output_dir / f"{pdf.stem}_pt_refinado.md"
            output_refined_pdf = cfg.output_dir / f"{pdf.stem}_pt_refinado.pdf"
            refine_markdown_file(
                input_path=md_path,
                output_path=output_refined,
                backend=refine_backend,
                cfg=cfg,
                logger=logger,
            )
            markdown_to_pdf(
                markdown_text=read_text(output_refined),
                output_path=output_refined_pdf,
                font_dir=cfg.font_dir,
                title_size=cfg.pdf_title_font_size,
                heading_size=cfg.pdf_heading_font_size,
                body_size=cfg.pdf_body_font_size,
                logger=logger,
            )


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

    for md in md_files:
        stem = md.stem.replace("_pt", "")
        output_md = cfg.output_dir / f"{stem}_pt_refinado.md"
        output_pdf = cfg.output_dir / f"{stem}_pt_refinado.pdf"
        refine_markdown_file(
            input_path=md,
            output_path=output_md,
            backend=backend,
            cfg=cfg,
            logger=logger,
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
    cfg = AppConfig()
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
