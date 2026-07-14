from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tradutor.glossary_audit import is_probably_portuguese_alias


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


def _remove_values(term: dict[str, Any], field: str, values: list[str]) -> bool:
    current = _as_list(term.get(field))
    remove = {value.casefold() for value in values}
    kept = [value for value in current if value.casefold() not in remove]
    if kept == current:
        return False
    term[field] = kept
    return True


def _append_bad_aliases(term: dict[str, Any], values: list[str]) -> bool:
    changed = _remove_values(term, "allowed_target_aliases", values)
    changed = _append_unique(term, "bad_aliases", values) or changed
    return changed


def _move_portuguese_source_aliases(term: dict[str, Any]) -> bool:
    source_aliases = _as_list(term.get("source_aliases") or term.get("aliases"))
    if not source_aliases:
        return False

    key_norm = str(term.get("key", "")).strip().casefold()
    pt_norm = str(term.get("pt", "")).strip().casefold()
    bad_norm = {value.casefold() for value in _as_list(term.get("bad_aliases"))}
    kept: list[str] = []
    moved_allowed: list[str] = []
    changed = False
    for alias in source_aliases:
        alias_norm = alias.strip().casefold()
        if not is_probably_portuguese_alias(alias):
            kept.append(alias)
            continue
        changed = True
        if alias_norm in {key_norm, pt_norm} or alias_norm in bad_norm:
            continue
        moved_allowed.append(alias)

    changed = _set_source_aliases(term, kept) or changed
    changed = _append_unique(term, "allowed_target_aliases", moved_allowed) or changed
    return changed


def _remove_redundant_source_aliases(term: dict[str, Any]) -> bool:
    source_aliases = _as_list(term.get("source_aliases") or term.get("aliases"))
    if not source_aliases:
        return False
    redundant = {
        str(term.get("key", "")).strip().casefold(),
        str(term.get("pt", "")).strip().casefold(),
    }
    kept = [
        alias for alias in source_aliases if alias.strip().casefold() not in redundant
    ]
    if kept == source_aliases:
        return False
    return _set_source_aliases(term, kept)


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
        changed = (
            _append_unique(term, "bad_aliases", ["Discípulos de Vicius"]) or changed
        )
        mark(changed)

    term = _find_term(terms, "Forbidden Words Clan")
    if term:
        changed = _set_source_aliases(term, ["Kurosaga", "Kurosaga Clan"])
        changed = _append_unique(term, "bad_aliases", ["Clã Kurosaga"]) or changed
        mark(changed)

    term = _find_term(terms, "Myeow")
    if term:
        changed = _set_source_aliases(term, ["Meow", "Myaah", "Mya-a-ah"])
        changed = (
            _append_unique(term, "bad_aliases", ["meow", "Meow", "myeow", "Myeow"])
            or changed
        )
        mark(changed)

    term = _find_term(terms, "Wildly Beautiful Emperor")
    if term:
        changed = _set_source_aliases(
            term, ["Beautiful Wild Emperor", "Madly Beautiful Emperor"]
        )
        changed = _append_unique(term, "allowed_target_aliases", ["Zine"]) or changed
        mark(changed)

    term = _find_term(terms, "Belzegea")
    if term:
        changed = _set_source_aliases(term, ["Fly Guy"])
        changed = (
            _append_unique(term, "allowed_target_aliases", ["Senhor das Moscas"])
            or changed
        )
        mark(changed)

    term = _find_term(terms, "Anael")
    if term:
        changed = _set_source_aliases(term, ["Mistress Anael"])
        mark(changed)

    term = _find_term(terms, "Erika Anaorbael")
    if term:
        changed = _set_source_aliases(term, ["Erika"])
        mark(changed)

    term = _find_term(terms, "Lokiella")
    if term:
        changed = _set_source_aliases(term, ["Loki-ella"])
        mark(changed)

    term = _find_term(terms, "Four Holy Elders")
    if term:
        changed = _set_source_aliases(term, ["Four Holy Elder", "Holy Elders"])
        changed = (
            _append_bad_aliases(
                term,
                [
                    "Quatro Anciãos Sagrados",
                    "Quatro Anciões Sagrados",
                    "Quatro Santos Anciãos",
                    "Quatro Anciãos Santos",
                    "Quatro Sábios",
                ],
            )
            or changed
        )
        mark(changed)

    term = _find_term(terms, "Monster Slayer King")
    if term:
        changed = _set_source_aliases(term, ["King of Monster Slayers"])
        changed = (
            _append_bad_aliases(
                term, ["Rei Matador de Monstros", "Rei Exterminador de Monstros"]
            )
            or changed
        )
        mark(changed)

    term = _find_term(terms, "Monster Slayer King of Ulza")
    if term:
        changed = _set_source_aliases(term, ["King of Monster Slayers of Ulza"])
        changed = (
            _append_bad_aliases(
                term,
                [
                    "Rei Matador de Monstros de Ulza",
                    "Rei Exterminador de Monstros de Ulza",
                ],
            )
            or changed
        )
        mark(changed)

    term = _find_term(terms, "Monster Slayer Knights")
    if term:
        changed = _set_source_aliases(
            term, ["Monster Slayer Knight Order", "Knights of Monster Slaying"]
        )
        changed = (
            _append_bad_aliases(
                term,
                [
                    "Cavaleiros Matadores de Monstros",
                    "Cavaleiros Exterminadores de Monstros",
                ],
            )
            or changed
        )
        mark(changed)

    term = _find_term(terms, "Sabre-Toothed Tigers")
    if term:
        changed = _set_source_aliases(
            term,
            [
                "Sabertooth Tigers",
                "Sabre-toothed Tigers",
                "Saber-Toothed Tigers",
                "Sabre Toothed Tigers",
            ],
        )
        changed = (
            _append_bad_aliases(
                term,
                [
                    "Tigres de Dente de Sabre",
                    "Tigres de Dente-de-Sabre",
                    "Tigres Dente de Sabre",
                    "Tigres-dentes-de-sabre",
                ],
            )
            or changed
        )
        mark(changed)

    term = _find_term(terms, "Dragon-Eye Cup")
    if term:
        changed = _append_bad_aliases(term, ["Cálice Olho de Dragão"])
        mark(changed)

    term = _find_term(terms, "Mils Ruins")
    if term:
        changed = _set_source_aliases(term, ["Ancient Dragon Ruins"])
        changed = _append_bad_aliases(term, ["Ruínas do Dragão Antigo"]) or changed
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

    for term in terms:
        changed = _move_portuguese_source_aliases(term)
        changed = _remove_redundant_source_aliases(term) or changed
        mark(changed)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate glossary search aliases from forbidden target forms."
    )
    parser.add_argument("path", help="Glossary JSON path.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes. Without this, runs as dry-run.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = migrate(data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
