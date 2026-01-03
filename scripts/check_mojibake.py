from __future__ import annotations

import sys
from pathlib import Path


MOJIBAKE_TOKENS = [
    "ÔÇ£",
    "ÔÇØ",
    "ÔÇª",
    "Ã¢â‚¬",
    "├Ç",
    "├┐",
]


def main(argv: list[str]) -> int:
    paths = [Path(path) for path in argv[1:]]
    if not paths:
        return 0

    failed = False
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            sys.stderr.write(f"Arquivo nao e UTF-8: {path} ({exc})\n")
            failed = True
            continue
        for token in MOJIBAKE_TOKENS:
            if token in content:
                sys.stderr.write(
                    f"Encontrado mojibake '{token}' em {path}\n"
                )
                failed = True
                break

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
