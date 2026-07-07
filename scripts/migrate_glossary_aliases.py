from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _terms_doc(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("terms"), list):
        return data["terms"]
    if isinstance(data, list):
        return data
    raise ValueError("Glossary must be a JSON object with 'terms' or a list of terms.")


def _find_term(terms: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    key_l = key.casefold()
    for term in terms:
        if str(term.get("key", "")).casefold() == key_l:
            return term
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = value.strip()
        if not clean:
            continue
        marker = clean.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        out.append(clean)
    return out


def _set_source_aliases(term: dict[str, Any], aliases: list[str]) -> bool:
    aliases = _unique(aliases)
    changed = term.get("source_aliases") != aliases or term.get("aliases") != aliases
    term["source_aliases"] = aliases
    # Legacy compatibility: aliases now mirrors source-side search aliases only.
    term["aliases"] = aliases
    return changed


def _append_unique(term: dict[str, Any], field: str, values: list[str]) -> bool:
    current = _as_list(term.get(field))
    merged = _unique(current + values)
    if current == merged:
        return False
    term[field] = merged
    return True


def migrate(data: Any) -> dict[str, int]:
    terms = _terms_doc(data)
    summary: dict[str, int] = {
        "terms_updated": 0,
        "terms_added": 0,
    }

    def mark(changed: bool) -> None:
        if changed:
            summary["terms_updated"] += 1

    term = _find_term(terms, "Children of Vicius")
    if term:
        changed = _set_source_aliases(term, ["Vicius's Disciples"])
        changed = _append_unique(term, "bad_aliases", ["Discípulos de Vicius"]) or changed
        mark(changed)

    term = _find_term(terms, "Forbidden Words Clan")
    if term:
        changed = _set_source_aliases(term, ["Kurosaga", "Kurosaga Clan"])
        changed = _append_unique(term, "bad_aliases", ["Clã Kurosaga"]) or changed
        mark(changed)

    term = _find_term(terms, "Myeow")
    if term:
        changed = _set_source_aliases(term, ["Meow", "Myaah", "Mya-a-ah"])
        changed = _append_unique(term, "bad_aliases", ["meow", "Meow", "myeow", "Myeow"]) or changed
        mark(changed)

    term = _find_term(terms, "Wildly Beautiful Emperor")
    if term:
        changed = _set_source_aliases(term, ["Beautiful Wild Emperor", "Madly Beautiful Emperor"])
        changed = _append_unique(term, "allowed_target_aliases", ["Zine"]) or changed
        mark(changed)

    if not _find_term(terms, "Goddess-chin"):
        terms.append(
            {
                "key": "Goddess-chin",
                "pt": "Deusazinha",
                "category": "apelido",
                "notes": "Apelido informal/pejorativo usado para a Deusa Vicius. Evitar a forma híbrida Deusa-chin.",
                "source_aliases": ["Goddess-chin", "Goddess Vicius-chin"],
                "aliases": ["Goddess-chin", "Goddess Vicius-chin"],
                "bad_aliases": ["Deusa-chin"],
                "source": "revisao_tecnica",
                "locked": True,
            }
        )
        summary["terms_added"] += 1

    character_aliases = {
        "Ikusaba Asagi": ["Asagi Ikusaba", "Asagi", "Asagi-san", "Ikusaba"],
        "Kashima Kobato": ["Kobato Kashima", "Kashima", "Kashima-san", "Kobato"],
        "Takao Hijiri": ["Hijiri Takao", "Hijiri", "Hijiri-san"],
        "Sogou Ayaka": ["Ayaka Sogou", "Sogou", "Sogou-san", "Ayaka"],
        "Takao Itsuki": ["Itsuki Takao", "Itsuki", "Itsuki-san"],
    }
    for key, aliases in character_aliases.items():
        term = _find_term(terms, key)
        if not term:
            continue
        changed = _set_source_aliases(term, aliases)
        if term.get("gender") != "feminino":
            term["gender"] = "feminino"
            changed = True
        if term.get("type") != "personagem":
            term["type"] = "personagem"
            changed = True
        mark(changed)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Separate glossary search aliases from forbidden target forms.")
    parser.add_argument("path", help="Glossary JSON path.")
    parser.add_argument("--write", action="store_true", help="Write changes. Without this, runs as dry-run.")
    args = parser.parse_args()

    path = Path(args.path)
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = migrate(data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
