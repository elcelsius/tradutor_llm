from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


def is_fenced_code_boundary(line: str) -> bool:
    """
    Detects start or end of a fenced code block (triple backticks).
    Leading whitespace is ignored.
    """
    return line.lstrip().startswith("```")


def is_heading(line: str) -> bool:
    """Returns True when the line is a Markdown heading."""
    return bool(re.match(r"^\s*#{1,6}\s", line))


def is_list_item(line: str) -> bool:
    """Returns True for unordered or ordered list markers."""
    return bool(re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line))


def is_blockquote(line: str) -> bool:
    """Returns True for blockquote lines."""
    return bool(re.match(r"^\s*>", line))


def is_table_line(line: str) -> bool:
    """
    Heuristic to identify Markdown table rows.
    Looks for pipes and common table patterns; intended to avoid joining them.
    """
    if "|" not in line:
        return False
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("|") or stripped.endswith("|"):
        return True
    if " | " in line:
        return True
    if re.match(r"^\s*:?[-]{2,}\s*\|", stripped):
        return True
    return False


def is_horizontal_rule(line: str) -> bool:
    """Detects horizontal rules like --- or *** to keep them isolated."""
    stripped = line.strip()
    return bool(re.match(r"^([-*_]\s*){3,}$", stripped))


def is_structural_line(line: str) -> bool:
    """
    Checks for lines that should not be merged with surrounding prose.
    Includes headings, lists, blockquotes, tables and horizontal rules.
    """
    return any(
        check(line)
        for check in (is_heading, is_list_item, is_blockquote, is_table_line, is_horizontal_rule)
    )


def normalize_md_paragraphs(text: str) -> str:
    """
    Normaliza parágrafos com lógica estrita para não colar Títulos e Diálogos.
    """
    lines = text.splitlines()
    output = []
    current_paragraph = []

    def flush_paragraph():
        if current_paragraph:
            # Junta o buffer. Remove espaços duplos que possam surgir.
            full_text = " ".join(current_paragraph).strip()
            output.append(full_text)
            current_paragraph.clear()

    # Regex Compilados para Performance e Precisão

    # 1. Detecta início de diálogo (Aspas curvas/retas ou travessão)
    # Ex: "Olá", “Olá”, — Olá, - Olá
    RE_DIALOGUE_START = re.compile(r'^\s*(?:[“"\'\-] |[—–])')

    # 2. Detecta Títulos comuns em Novels (Capítulo X, Prólogo, etc ou CAIXA ALTA CURTA)
    RE_TITLE_KEYWORD = re.compile(r'^\s*(?:capítulo|prólogo|epílogo|parte|volume|livro|interlúdio)\b', re.IGNORECASE)
    RE_ALL_CAPS_SHORT = re.compile(r'^\s*[A-Z0-9\W]{3,50}\s*$')  # Linhas curtas em CAPS LOCK

    # 3. Detecta fim de sentença forte (., !, ?, ", ”)
    RE_SENTENCE_END = re.compile(r'(?:[.?!]|”|")\s*$')

    # 4. Detecta fim com travessão (interrupção)
    RE_DASH_END = re.compile(r'[—–-]\s*$')

    for raw_line in lines:
        stripped = raw_line.strip()

        # A. Se linha vazia ou estrutural (Markdown), flush imediato.
        if not stripped or is_structural_line(raw_line):
            flush_paragraph()
            output.append(raw_line.rstrip())  # Mantém estrutura original
            continue

        # Se não temos nada no buffer, inicia novo parágrafo
        if not current_paragraph:
            current_paragraph.append(stripped)
            continue

        prev_line = current_paragraph[-1]

        # --- LÓGICA DE DECISÃO (Devemos Juntar?) ---
        should_merge = True

        # REGRA 1: Títulos (Prioridade Máxima)
        # Se parece título (palavra chave ou caixa alta curta), NÃO junta.
        if RE_TITLE_KEYWORD.match(stripped) or RE_ALL_CAPS_SHORT.match(stripped):
            should_merge = False

        # REGRA 2: Diálogos
        # Se a linha atual começa com aspas ou travessão, é fala nova. NÃO junta.
        elif RE_DIALOGUE_START.match(stripped):
            should_merge = False

        # REGRA 3: Pontuação Forte + Letra Maiúscula
        # Se a anterior acabou em ponto e a atual começa com Maiúscula, é novo parágrafo.
        # (Exceção: se a atual começa com minúscula, é continuação errada do OCR, então junta).
        elif RE_SENTENCE_END.search(prev_line) and stripped[0].isupper():
            should_merge = False

        # REGRA 4: Travessão de Interrupção
        # Se a anterior acabou em "—" e a atual começa com Maiúscula ou Aspas, NÃO junta.
        elif RE_DASH_END.search(prev_line) and (stripped[0].isupper() or stripped.startswith(('“', '"'))):
            should_merge = False

        # --- APLICAÇÃO ---
        if should_merge:
            # Tratamento especial para palavras hifenizadas quebradas (ex: "comuni-\ncação")
            if prev_line.endswith('-') and not prev_line.endswith(' -'):
                current_paragraph[-1] = prev_line[:-1] + stripped
            else:
                current_paragraph.append(stripped)
        else:
            # Não junta -> Flush do anterior e começa novo
            flush_paragraph()
            current_paragraph.append(stripped)

    flush_paragraph()
    return "\n".join(output)


def _default_output_path(input_path: Path) -> Path:
    """Builds the default output path by adding '_norm' before the extension."""
    suffix = input_path.suffix
    name = f"{input_path.stem}_norm{suffix}" if suffix else f"{input_path.stem}_norm"
    return input_path.with_name(name)


def normalize_file(input_path: Path, output_path: Path) -> Path:
    """
    Reads a Markdown file, normalizes paragraphs, and writes the result.

    Returns the path to the written file.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = input_path.read_text(encoding="utf-8")
    normalized = normalize_md_paragraphs(content)
    output_path.write_text(normalized, encoding="utf-8")
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove unwanted line breaks inside Markdown paragraphs while preserving structure."
    )
    parser.add_argument("--input", required=True, help="Path to the Markdown file to normalize.")
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to adding '_norm' before the extension.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    written_path = normalize_file(input_path, output_path)
    print(written_path)


if __name__ == "__main__":
    # Example usage:
    # python desquebrar.py --input "saida/MEU_LIVRO_pt.md"
    # python desquebrar.py --input "saida/MEU_LIVRO_pt.md" --output "saida/MEU_LIVRO_pt_limpo.md"
    main()
