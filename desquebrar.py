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
    Normaliza parágrafos de Markdown usando heurísticas mais estritas para evitar colagem indevida.

    Regras de decisão para juntar (merge) current_line com prev_line:
    1) Estruturais: nunca juntam (heading/lista/tabela/código/etc.).
    2) Diálogo/Título/Travessão/Pontuação (não juntar):
       - Se current_line começa com aspas (“, ") ou travessão (—, –, -), nunca junta.
       - Se prev_line termina com pontuação final forte (. ! ? ” ") e current_line começa com maiúscula, nunca junta.
       - Se prev_line termina com travessão (—, –, --) e current_line começa com aspas ou maiúscula, nunca junta.
       - Se current_line for título em maiúsculas curto (<40 chars, sem pontuação) ou começar com Capítulo/Prólogo/Epílogo/Parte/Livro/Volume (<100 chars), nunca junta.
       - Se prev_line for toda maiúscula curta (<100 chars) e a atual começa com maiúscula, não junta.
       - Se prev_line termina com ponto (.) e current_line começa com aspas, não junta.
    3) Continuação (único caso que junta):
       - Se current_line começa com minúscula, junta; OU
       - Se prev_line não termina com pontuação terminal (. ! ? … ” "), junta.
    Caso contrário, novo parágrafo.
    """
    lines = text.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    in_code_block = False

    dialogue_prefix = re.compile(r'^\s*(?:—|–|-|"|“)')  # marcadores de diálogo
    starts_lower = re.compile(r'^\s*[a-záàâãéèêíïóôõöúçñ]')  # minúsculas comuns PT
    starts_upper = re.compile(r'^\s*[A-ZÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]')  # maiúsculas comuns PT
    ends_with_terminal = re.compile(r'[\.!?…”"]\s*$')
    ends_with_dash = re.compile(r'(—|–|--)\s*$')
    title_keyword = re.compile(r'^\s*(cap[íi]tulo|pr[óo]logo|ep[íi]logo|parte|livro|volume)\b', re.IGNORECASE)

    def flush_paragraph() -> None:
        if paragraph:
            output.append(" ".join(paragraph).strip())
            paragraph.clear()

    for raw_line in lines:
        if is_fenced_code_boundary(raw_line):
            flush_paragraph()
            output.append(raw_line.rstrip("\n"))
            in_code_block = not in_code_block
            continue

        if in_code_block:
            output.append(raw_line.rstrip("\n"))
            continue

        stripped = raw_line.strip()
        if stripped == "":
            flush_paragraph()
            output.append("")
            continue

        if is_structural_line(raw_line):
            flush_paragraph()
            output.append(raw_line.rstrip("\n"))
            continue

        # Heurísticas de junção
        if not paragraph:
            paragraph.append(stripped)
            continue

        prev_line = paragraph[-1]
        join = False

        # Guardrails: separação obrigatória
        # Título por palavra-chave explícita (<80 chars)
        if title_keyword.match(stripped) and len(stripped) < 80:
            flush_paragraph()
            output.append(stripped)
            paragraph.clear()
            continue

        prev_upper_short = len(prev_line) < 100 and prev_line.isupper()

        # Proteção de diálogo (aspas iniciais)
        if dialogue_prefix.match(stripped):
            join = False
        # Proteção de travessão final: só junta se atual começa com minúscula
        elif ends_with_dash.search(prev_line):
            if starts_lower.match(stripped):
                join = True
            else:
                join = False
        # Proteção de pontuação forte + próxima maiúscula
        elif ends_with_terminal.search(prev_line) and starts_upper.match(stripped):
            join = False
        # Título em maiúsculas curto sem pontuação => não junta
        elif len(stripped) < 40 and stripped.isupper() and not ends_with_terminal.search(stripped):
            join = False
        # Prev toda maiúscula curta + atual maiúscula => não junta
        elif prev_upper_short and starts_upper.match(stripped):
            join = False
        # Prev termina com ponto e atual começa com aspas => não junta
        elif prev_line.rstrip().endswith(".") and dialogue_prefix.match(stripped):
            join = False
        else:
            # Único caso que junta: começa com minúscula OU anterior sem pontuação terminal
            if starts_lower.match(stripped):
                join = True
            elif not ends_with_terminal.search(prev_line):
                join = True
            else:
                join = False

        if join:
            paragraph.append(stripped)
        else:
            flush_paragraph()
            paragraph.append(stripped)

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
