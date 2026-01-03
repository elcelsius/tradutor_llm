from __future__ import annotations

from pathlib import Path

import pytest


MOJIBAKE_TOKENS = [
    "\u00d4\u00c7\u00a3",  # mojibake for opening curly quote
    "\u00d4\u00c7\u00d8",  # mojibake for closing curly quote
    "\u00d4\u00c7\u00aa",  # mojibake for ellipsis
    "\u251c\u00c7",  # mojibake fragment
    "\u251c\u2510",  # mojibake fragment
    "\u00c3\u00a2\u20ac",  # mojibake fragment
    "\u00c3\u00a2\u20ac\u201c",  # mojibake fragment
]


def _iter_source_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    sources: list[Path] = []
    for rel in ("tradutor", "tests"):
        sources.extend((root / rel).rglob("*.py"))
    return sources


@pytest.mark.parametrize("path", _iter_source_files())
def test_source_files_are_utf8_and_free_of_mojibake(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(f"Arquivo nao e UTF-8: {path} ({exc})")
    for token in MOJIBAKE_TOKENS:
        assert token not in content, f"Encontrado mojibake '{token}' em {path}"
