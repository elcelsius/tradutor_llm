"""
Configurações centrais do pipeline de tradução e refine.

Mantém valores padrão em um único lugar para facilitar manutenção e leitura.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


BackendType = Literal["ollama", "gemini"]


@dataclass(frozen=True)
class AppConfig:
    """Valores padrão para todo o pipeline."""

    # Diretórios padrão
    data_dir: Path = Path("data")
    output_dir: Path = Path("saida")
    font_dir: Path = Path(".cache/fonts")

    # Modelos
    translate_backend: BackendType = "ollama"
    translate_model: str = "qwen3:14b-q4_K_M"
    refine_backend: BackendType = "ollama"
    refine_model: str = "gemma3-gaia-ptbr-4b:q4_k_m"

    # Temperaturas
    translate_temperature: float = 0.15
    refine_temperature: float = 0.30

    # Chunk sizes
    translate_chunk_chars: int = 3800
    refine_chunk_chars: int = 10_000

    # Tentativas e backoff
    max_retries: int = 3
    initial_backoff: float = 1.5
    backoff_factor: float = 1.8

    # Timeouts
    request_timeout: int = 120

    # PDF
    pdf_title_font_size: int = 16
    pdf_heading_font_size: int = 13
    pdf_body_font_size: int = 11


def ensure_paths(cfg: AppConfig) -> None:
    """Garante que os diretórios principais existam."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.font_dir.mkdir(parents=True, exist_ok=True)
