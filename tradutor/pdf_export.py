"""
Exporta Markdown simplificado para PDF com tipografia local (Windows) e layout otimizado para leitura digital.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

try:
    import pyphen
except Exception:  # pragma: no cover - opcional
    pyphen = None

from .utils import ensure_dir


def _register_font(logger: logging.Logger) -> str:
    """
    Seleciona a primeira fonte disponível na ordem de preferência, sem downloads.
    Fallback: Helvetica (built-in do ReportLab).
    """
    candidates = [
        ("Aptos", Path("C:/Windows/Fonts/Aptos.ttf")),
        ("Aptos Display", Path("C:/Windows/Fonts/AptosDisplay.ttf")),
        ("SegoeUI", Path("C:/Windows/Fonts/segoeui.ttf")),
        ("Calibri", Path("C:/Windows/Fonts/Calibri.ttf")),
        ("Arial", Path("C:/Windows/Fonts/Arial.ttf")),
    ]
    for name, path in candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont(name, str(path)))
                logger.info("Usando fonte local para PDF: %s", path)
                return name
            except Exception as exc:  # pragma: no cover
                logger.warning("Falha ao registrar fonte %s: %s", path, exc)
                continue
    logger.warning("Nenhuma fonte preferencial encontrada; usando Helvetica (built-in).")
    return "Helvetica"


def _build_styles(font_name: str) -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    body_leading = 11.5 * 1.35
    dialogue_leading = 11.5 * 1.25
    hyphenator = None
    if pyphen is not None:
        try:
            hyphenator = pyphen.Pyphen(lang="pt_BR")
        except Exception:
            hyphenator = None

    styles.add(
        ParagraphStyle(
            name="Body",
            fontName=font_name,
            fontSize=11.5,
            leading=body_leading,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            hyphenation=hyphenator,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Dialogue",
            fontName=font_name,
            fontSize=11.5,
            leading=dialogue_leading,
            alignment=TA_LEFT,
            firstLineIndent=0,
            leftIndent=0,
            spaceAfter=4,
            hyphenation=hyphenator,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading1",
            fontName=font_name,
            fontSize=18,
            leading=18 * 1.2,
            spaceAfter=10,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading2",
            fontName=font_name,
            fontSize=14,
            leading=14 * 1.2,
            spaceAfter=8,
            alignment=TA_LEFT,
        )
    )
    return styles


def _is_dialogue_line(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("— ") or stripped.startswith("- ") or stripped.startswith("– ")


def _build_story(lines: Iterable[str], styles: dict[str, ParagraphStyle]):
    story = []
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 6))
            continue
        if stripped.startswith("## "):
            text = escape(stripped[3:].strip())
            story.append(Paragraph(text, styles["Heading2"]))
            continue
        if stripped.startswith("# "):
            text = escape(stripped[2:].strip())
            story.append(Paragraph(text, styles["Heading1"]))
            continue
        if _is_dialogue_line(stripped):
            text = escape(stripped)
            story.append(Paragraph(text, styles["Dialogue"]))
        else:
            text = escape(stripped)
            story.append(Paragraph(text, styles["Body"]))
    return story


def markdown_to_pdf(
    markdown_text: str,
    output_path: Path,
    font_dir: Path,
    title_size: int,
    heading_size: int,
    body_size: int,
    logger: logging.Logger,
) -> None:
    """Converte Markdown simplificado para PDF com layout otimizado para leitura digital."""
    ensure_dir(output_path.parent)
    font_name = _register_font(logger)
    styles = _build_styles(font_name)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=52,
        bottomMargin=52,
    )
    lines = markdown_text.splitlines()
    story = _build_story(lines, styles)
    doc.build(story)
    logger.info("PDF gerado: %s", output_path)
