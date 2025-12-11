"""
Configurações centrais do pipeline de tradução e refine.

Mantém valores padrão em um único lugar para facilitar manutenção e leitura.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


BackendType = Literal["ollama", "gemini"]
DEFAULT_CONFIG_PATHS = (Path("config.yaml"), Path("config.yml"))
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    """Valores padrão para todo o pipeline."""

    # Diretórios padrão
    data_dir: Path = Path("data")
    output_dir: Path = Path("saida")
    font_dir: Path = Path(".cache/fonts")

    # Modelos
    translate_backend: BackendType = "ollama"
    translate_model: str = "brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16"
    refine_backend: BackendType = "ollama"
    refine_model: str = "cnmoro/gemma3-gaia-ptbr-4b:q4_k_m"
    dump_chunks: bool = False

    # Temperaturas
    translate_temperature: float = 0.15
    refine_temperature: float = 0.30
    translate_repeat_penalty: float = 1.1
    refine_repeat_penalty: float | None = None

    # Chunk sizes
    translate_chunk_chars: int = 2400
    refine_chunk_chars: int = 2400

    # Tentativas e backoff
    max_retries: int = 3
    initial_backoff: float = 1.5
    backoff_factor: float = 1.8

    # Timeouts
    request_timeout: int = 60

    # PDF
    pdf_title_font_size: int = 16
    pdf_heading_font_size: int = 13
    pdf_body_font_size: int = 11


def ensure_paths(cfg: AppConfig) -> None:
    """Garante que os diretórios principais existam."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.font_dir.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Carrega configurações a partir de YAML, com fallback para valores padrão.
    """
    base = AppConfig()

    path: Path | None = None
    if config_path:
        candidate = Path(config_path)
        if candidate.exists():
            path = candidate
    else:
        for candidate in DEFAULT_CONFIG_PATHS:
            if candidate.exists():
                path = candidate
                break

    if path is None:
        return base

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return base
    except Exception as exc:  # pragma: no cover - I/O edge case
        log.warning("Falha ao ler config %s; usando defaults. Erro: %s", path, exc)
        return base

    if not isinstance(data, dict):
        log.warning("Config %s tem formato inesperado; usando defaults.", path)
        return base

    overrides = {}
    for key, value in data.items():
        if key not in base.__dict__:
            continue
        if key.endswith("_dir"):
            overrides[key] = Path(value)
        else:
            overrides[key] = value

    merged = {**base.__dict__, **overrides}
    return AppConfig(**merged)
