from __future__ import annotations

from pathlib import Path

import pytest


MOJIBAKE_TOKENS = [
    "\u00d4\u00c7\u00a3",
    "\u00d4\u00c7\u00d8",
    "\u00d4\u00c7\u00aa",
    "\u00c3\u00a2\u20ac",
    "\u251c\u00c7",
    "\u251c\u2510",
]


def _iter_source_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    sources: set[Path] = set()
    patterns = ("*.py", "*.md", "*.json")
    for rel in ("tradutor", "tests"):
        base = root / rel
        for pattern in patterns:
            sources.update(base.rglob(pattern))
    return sorted(sources)


@pytest.mark.parametrize("path", _iter_source_files())
def test_source_files_are_utf8_and_free_of_mojibake(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(f"Arquivo nao e UTF-8: {path} ({exc})")
    for token in MOJIBAKE_TOKENS:
        assert token not in content, f"Encontrado mojibake '{token}' em {path}"
