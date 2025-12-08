"""
Exporta Markdown simples para PDF com suporte a Unicode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import requests
from fpdf import FPDF

from .utils import ensure_dir


DEJAVU_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"


def ensure_font(font_dir: Path, logger: logging.Logger) -> Path:
    """Baixa DejaVuSans.ttf se não existir."""
    ensure_dir(font_dir)
    font_path = font_dir / "DejaVuSans.ttf"
    if font_path.exists():
        return font_path
    logger.info("Baixando fonte DejaVuSans.ttf para %s", font_path)
    resp = requests.get(DEJAVU_URL, timeout=60)
    resp.raise_for_status()
    font_path.write_bytes(resp.content)
    return font_path


class SimplePDF(FPDF):
    """Conversão básica de Markdown para PDF."""

    def __init__(self, font_path: Path, title_size: int, heading_size: int, body_size: int):
        super().__init__()
        self.font_path = font_path
        self.title_size = title_size
        self.heading_size = heading_size
        self.body_size = body_size

    def header(self) -> None:  # type: ignore[override]
        pass  # sem cabeçalho

    def footer(self) -> None:  # type: ignore[override]
        pass  # sem rodapé

    def add_markdown(self, lines: Iterable[str]) -> None:
        self.add_page()
        self.add_font("DejaVu", "", str(self.font_path), uni=True)
        self.set_font("DejaVu", size=self.body_size)

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## "):
                self.set_font("DejaVu", "B", self.heading_size)
                self.multi_cell(0, 8, stripped[3:].strip())
                self.ln(2)
                self.set_font("DejaVu", size=self.body_size)
            elif stripped.startswith("# "):
                self.set_font("DejaVu", "B", self.title_size)
                self.multi_cell(0, 10, stripped[2:].strip())
                self.ln(3)
                self.set_font("DejaVu", size=self.body_size)
            else:
                self.multi_cell(0, 6, stripped)
                self.ln(2)


def markdown_to_pdf(
    markdown_text: str,
    output_path: Path,
    font_dir: Path,
    title_size: int,
    heading_size: int,
    body_size: int,
    logger: logging.Logger,
) -> None:
    """Converte Markdown simplificado para PDF."""
    ensure_dir(output_path.parent)
    font_path = ensure_font(font_dir, logger)
    pdf = SimplePDF(font_path=font_path, title_size=title_size, heading_size=heading_size, body_size=body_size)
    lines = markdown_text.splitlines()
    pdf.add_markdown(lines)
    pdf.output(str(output_path))
    logger.info("PDF gerado: %s", output_path)
