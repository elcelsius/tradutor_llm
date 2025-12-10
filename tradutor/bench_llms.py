
"""
Benchmark simples para comparar modelos Ollama na traducao usando o prompt do pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import requests

from tradutor.pdf_reader import extract_pdf_text
from tradutor.translate import build_translation_prompt


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


def _list_models_via_cli() -> list[str]:
    """
    Usa `ollama list` para obter modelos instalados. Retorna lista vazia em caso de falha.
    """
    cmd_json = ["ollama", "list", "--format", "json"]
    for cmd in (cmd_json, ["ollama", "list"]):
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=10
            )
        except Exception:
            continue
        output = result.stdout.strip()
        if not output:
            continue
        try:
            data = json.loads(output)
            names = [item["name"] for item in data if isinstance(item, dict) and "name" in item]
            if names:
                return names
        except Exception:
            pass
        names: list[str] = []
        for line in output.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("name"):
                continue
            parts = line.split()
            if parts:
                names.append(parts[0])
        if names:
            return names
    return []


def _list_models_via_api(endpoint: str) -> set[str]:
    """
    Obtem a lista de modelos instalados no Ollama a partir de /api/tags.
    Se falhar, retorna conjunto vazio para nao bloquear a execucao.
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
        return {m["name"] for m in data.get("models", []) if "name" in m}
    except Exception:
        return set()


def list_installed_models(endpoint: str) -> list[str]:
    """
    Descobre modelos usando `ollama list` (preferencial) ou /api/tags.
    """
    models = _list_models_via_cli()
    if models:
        return models
    return sorted(_list_models_via_api(endpoint))


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

    installed = list_installed_models(args.endpoint)
    if args.models:
        models = args.models
        if installed:
            missing = [m for m in models if m not in installed]
            available = [m for m in models if m in installed]
            if missing:
                print(f"Atencao: ignorando modelos nao instalados: {', '.join(missing)}")
            if available:
                models = available
            elif missing:
                raise SystemExit("Nenhum dos modelos informados esta instalado segundo o Ollama.")
    else:
        models = installed
        if not models:
            raise SystemExit(
                "Nenhum modelo Ollama foi encontrado. Rode `ollama list` para confirmar as instalacoes ou use --models."
            )

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
