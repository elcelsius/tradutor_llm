from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradutor.post_translation_review import load_glossary_terms, load_sections, review_translation_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply deterministic post-translation review.")
    parser.add_argument("--input", required=True, help="Translated Markdown file.")
    parser.add_argument("--output", required=True, help="Reviewed Markdown output path.")
    parser.add_argument("--sections", help="sections.json from debug run.")
    parser.add_argument("--glossary", help="Manual glossary JSON.")
    parser.add_argument("--report", help="Optional JSON report path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    sections = load_sections(args.sections)
    terms = load_glossary_terms(args.glossary)
    text = input_path.read_text(encoding="utf-8")
    reviewed, report = review_translation_text(text, sections=sections, glossary_terms=terms)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(reviewed, encoding="utf-8")
    payload = report.to_dict()
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
