
"""
Benchmark simples para comparar modelos Ollama na traducao usando o prompt do pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import List

import requests

from tradutor.pdf_reader import extract_pdf_text
from tradutor.translate import build_translation_prompt

DEFAULT_MODELS: List[str] = [
    "cnmoro/gemma3-gaia-ptbr-4b:q4_k_m",
    "brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16",
    "huihui_ai/qwen3-abliterated:14b-q4_K_M",
    "qwen3:14b-q4_K_M",
    "cnmoro/Qwen2.5-0.5B-Portuguese-v2:fp16",
    "dolphin3:8b-llama3.1-q4_K_M",
]


def slugify_model(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def call_ollama(model: str, prompt: str, endpoint: str) -> tuple[str, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.15},
    }
    start = time.monotonic()
    try:
        resp = requests.post(endpoint, json=payload, timeout=300)
        elapsed = time.monotonic() - start
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao chamar Ollama para modelo '{model}' em {endpoint}: {exc}"
        ) from exc

    if "response" not in data:
        raise RuntimeError(f"Resposta invalida do Ollama para {model}: {json.dumps(data)[:200]}")
    return data["response"], elapsed


def list_installed_models(endpoint: str) -> set[str]:
    """
    Obtém a lista de modelos instalados no Ollama a partir de /api/tags.
    Se falhar, retorna conjunto vazio para não bloquear a execução.
    """
    tags_url = endpoint.rstrip("/")
    if tags_url.endswith("/generate"):
        tags_url = tags_url.rsplit("/", 1)[0] + "/tags"
    else:
        tags_url = tags_url + "/tags"
    try:
        resp = requests.get(tags_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = {m["name"] for m in data.get("models", []) if "name" in m}
        return models
    except Exception:
        return set()


def read_input(path: Path, max_chars: int) -> str:
    if path.suffix.lower() == ".pdf":
        text = extract_pdf_text(path, logger=None)
    else:
        text = path.read_text(encoding="utf-8")
    if max_chars > 0:
        text = text[:max_chars]
    return text.strip()


def write_model_output(out_dir: Path, slug: str, model: str, translated: str, elapsed: float, input_path: Path) -> str:
    model_slug = slugify_model(model)
    out_path = out_dir / f"{slug}_{model_slug}.md"
    header = [
        f"# Benchmark de traducao - {model}",
        f"- Modelo: {model}",
        f"- Arquivo de origem: {input_path}",
        f"- Tempo de resposta: {elapsed:.2f} s",
        "",
    ]
    out_path.write_text("\n".join(header) + "\n" + translated, encoding="utf-8")
    return out_path.name


def write_summary(
    out_dir: Path,
    slug: str,
    input_path: Path,
    used_chars: int,
    endpoint: str,
    rows: list[tuple[str, str, float]],
) -> None:
    lines = [
        f"# Resumo de benchmark de traducao - {slug}",
        "",
        f"- Arquivo de origem: {input_path}",
        f"- Caracteres usados: {used_chars}",
        f"- Endpoint: {endpoint}",
        "",
        "| Modelo | Arquivo de saida | Tempo (s) |",
        "|--------|------------------|-----------|",
    ]
    for model, fname, elapsed in rows:
        lines.append(f"| {model} | {fname} | {elapsed:.2f} |")
    (out_dir / f"resumo_{slug}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark de traducao com varios modelos Ollama.")
    parser.add_argument("--input", required=True, help="Arquivo de entrada em ingles (.txt, .md ou .pdf)")
    parser.add_argument("--models", nargs="*", help="Lista de modelos Ollama a usar")
    parser.add_argument("--max-chars", type=int, default=1500, help="Maximo de caracteres do texto de entrada")
    parser.add_argument("--out-dir", default="benchmark/traducao", help="Diretorio de saida para resultados")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:11434/api/generate",
        help="Endpoint do Ollama (default http://localhost:11434/api/generate)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {input_path}")

    models = args.models if args.models else DEFAULT_MODELS
    installed = list_installed_models(args.endpoint)
    if installed:
        filtered = [m for m in models if m in installed]
        missing = [m for m in models if m not in installed]
        if filtered:
            models = filtered
        if missing:
            print(f"Atenção: ignorando modelos não instalados: {', '.join(missing)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = read_input(input_path, max_chars=args.max_chars)
    prompt = build_translation_prompt(text)

    slug = input_path.stem.lower()
    rows: list[tuple[str, str, float]] = []

    for model in models:
        translated, elapsed = call_ollama(model=model, prompt=prompt, endpoint=args.endpoint)
        fname = write_model_output(out_dir, slug, model, translated, elapsed, input_path)
        rows.append((model, fname, elapsed))

    write_summary(out_dir, slug, input_path, len(text), args.endpoint, rows)


if __name__ == "__main__":
    main()
