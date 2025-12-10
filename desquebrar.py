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
    Collapses wrapped prose lines inside paragraphs while preserving Markdown structure.

    Rules:
    - Blank lines separate paragraphs and are kept as-is (multiple blanks are preserved).
    - Headings, lists, blockquotes, tables and code fences are not merged.
    - Code blocks fenced with ``` are left untouched.
    - Prose lines inside a paragraph are trimmed and joined with a single space.

    Example:
        Input:
            Line one broken
            in two parts.

            - Bullet
            Continuation

        Output:
            Line one broken in two parts.

            - Bullet
            Continuation
    """
    lines = text.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    in_code_block = False

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
