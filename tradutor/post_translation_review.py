from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .glossary_utils import build_glossary_state
from .translate import ensure_section_heading


@dataclass
class ReviewReport:
    heading_fixes: int = 0
    glossary_replacements: dict[str, int] = field(default_factory=dict)
    text_replacements: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading_fixes": self.heading_fixes,
            "glossary_replacements": self.glossary_replacements,
            "text_replacements": self.text_replacements,
        }


def _sub_word(
    text: str,
    pattern: str,
    repl: str,
    *,
    flags: int = re.IGNORECASE,
    preserve_case: bool = False,
) -> tuple[str, int]:
    if not preserve_case:
        return re.subn(pattern, repl, text, flags=flags)

    def _replace(match: re.Match[str]) -> str:
        found = match.group(0)
        if found.isupper():
            return repl.upper()
        if found[:1].isupper():
            return repl[:1].upper() + repl[1:]
        return repl

    return re.subn(pattern, _replace, text, flags=flags)


def _record(counter: dict[str, int], key: str, count: int) -> None:
    if count:
        counter[key] = counter.get(key, 0) + count


def _term_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _term_variants_for_article_fix(term: dict) -> list[str]:
    variants: list[str] = []
    for field in ("pt", "key"):
        value = str(term.get(field, "")).strip()
        if value:
            variants.append(value)
            variants.extend(token for token in value.split() if token[:1].isupper() and len(token) >= 3)
    variants.extend(_term_string_list(term.get("source_aliases")))
    variants.extend(_term_string_list(term.get("aliases")))
    variants.extend(_term_string_list(term.get("allowed_target_aliases")))
    out: list[str] = []
    seen: set[str] = set()
    for value in variants:
        clean = value.strip()
        marker = clean.casefold()
        if not clean or marker in seen:
            continue
        seen.add(marker)
        out.append(clean)
    return sorted(out, key=len, reverse=True)


def apply_editorial_replacements(text: str, report: ReviewReport | None = None) -> str:
    """Correcoes editoriais deterministicas e conservadoras para PT-BR."""
    rpt = report or ReviewReport()
    replacements = [
        (r"\bEu wish\b", "Quem me dera", False),
        (r"(?<=\w)—or\b", " — ou", False),
        (r"(?<=\w)—,", " —,", False),
        (r"\bbipede\b", "bípede", True),
        (r"\bsemi-deuses\b", "semideuses", True),
        (r"\bTh-the… y’re…", "El-eles…", False),
        (
            r"\bNão parece que a Deusa Vicius está manipulando Kashima Kobato, nem que ela está sendo\b",
            "Não parece que a Deusa Vicius está manipulando a Asagi, nem que ela está sendo",
            False,
        ),
        (r"\bDeusa-chin\b", "Deusazinha", False),
        (r"\bsus\s+AF\b", "suspeita pra caramba", False),
        (r"\bmeow\b", "miau", True),
        (r"\bskills\b", "habilidades", True),
        (r"\bskill\b", "habilidade", True),
    ]
    for pattern, repl, preserve_case in replacements:
        text, count = _sub_word(text, pattern, repl, preserve_case=preserve_case)
        _record(rpt.text_replacements, f"{pattern}->{repl}", count)
    return text


def apply_gendered_article_fixes(text: str, glossary_terms: list[dict], report: ReviewReport | None = None) -> str:
    """Corrige artigos masculinos antes de personagens femininas conhecidas."""
    rpt = report or ReviewReport()
    for term in glossary_terms:
        gender = str(term.get("gender", "")).strip().casefold()
        category = str(term.get("category") or term.get("type") or term.get("term_type") or "").strip().casefold()
        if not gender.startswith("femin") or "person" not in category and "personagem" not in category:
            continue
        for variant in _term_variants_for_article_fix(term):
            escaped = re.escape(variant)
            article_pairs = [
                (rf"\bo\s+{escaped}\b", f"a {variant}"),
                (rf"\bO\s+{escaped}\b", f"A {variant}"),
                (rf"\bdo\s+{escaped}\b", f"da {variant}"),
                (rf"\bDo\s+{escaped}\b", f"Da {variant}"),
                (rf"\bno\s+{escaped}\b", f"na {variant}"),
                (rf"\bNo\s+{escaped}\b", f"Na {variant}"),
                (rf"\bao\s+{escaped}\b", f"à {variant}"),
                (rf"\bAo\s+{escaped}\b", f"À {variant}"),
            ]
            for pattern, repl in article_pairs:
                text, count = re.subn(pattern, repl, text)
                _record(rpt.text_replacements, f"{pattern}->{repl}", count)
            text, count = re.subn(rf"\b({escaped}) como aliado\b", r"\1 como aliada", text)
            _record(rpt.text_replacements, f"{variant} como aliado->{variant} como aliada", count)
    return text


def apply_glossary_bad_aliases(text: str, glossary_terms: list[dict], report: ReviewReport | None = None) -> str:
    """Substitui apenas formas explicitamente proibidas pelo termo canonico."""
    rpt = report or ReviewReport()
    for term in glossary_terms:
        pt = str(term.get("pt", "")).strip()
        if not pt:
            continue
        bad_aliases = term.get("bad_aliases") or term.get("forbidden_aliases") or []
        if isinstance(bad_aliases, str):
            bad_aliases = [bad_aliases]
        if not isinstance(bad_aliases, list):
            continue
        for alias in bad_aliases:
            alias_s = str(alias).strip()
            if not alias_s or alias_s == pt:
                continue
            pattern = rf"(?<!\w){re.escape(alias_s)}(?!\w)"
            text, count = re.subn(pattern, pt, text, flags=re.IGNORECASE)
            _record(rpt.glossary_replacements, f"{alias_s}->{pt}", count)
    return text


def restore_headings_from_sections(text: str, sections: list[dict], report: ReviewReport | None = None) -> str:
    """Reinsere headings que existem no `sections.json` mas sumiram no texto traduzido."""
    rpt = report or ReviewReport()
    if not sections:
        return text
    paragraphs = re.split(r"\n\s*\n", text.strip())
    section_titles = [str(sec.get("title", "")) for sec in sections if str(sec.get("title", "")).strip()]
    section_titles = [title for title in section_titles if title.lower() != "full text"]
    for title in section_titles:
        heading, changed = ensure_section_heading("", title)
        if not heading or not changed:
            continue
        heading_plain = heading.lstrip("#").strip()
        heading_re = re.compile(rf"^#?\s*{re.escape(heading_plain)}\s*$", flags=re.IGNORECASE)
        if any(heading_re.match(p.strip().splitlines()[0].strip()) for p in paragraphs if p.strip()):
            continue
        insert_idx = _guess_heading_insert_index(paragraphs, title)
        if insert_idx is None:
            continue
        paragraphs.insert(insert_idx, heading)
        rpt.heading_fixes += 1
    return "\n\n".join(p.strip() for p in paragraphs if p.strip())


def _guess_heading_insert_index(paragraphs: list[str], source_title: str) -> int | None:
    title = source_title.strip().lower()
    if title.startswith("chapter 1"):
        return _find_first(paragraphs, (r"^Após o confronto", r"^Depois do confronto", r"^Após o Deathmatch"))
    if title.startswith("chapter 5"):
        return _find_first(paragraphs, (r"^Minha consciência voltou", r"^Minha mente voltou", r"^A minha consciência voltou"))
    if title == "prologue":
        return 0
    return None


def _find_first(paragraphs: list[str], patterns: tuple[str, ...]) -> int | None:
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    for idx, para in enumerate(paragraphs):
        first = para.strip().splitlines()[0].strip() if para.strip() else ""
        if any(rx.search(first) for rx in compiled):
            return idx
    return None


def review_translation_text(
    text: str,
    *,
    sections: list[dict] | None = None,
    glossary_terms: list[dict] | None = None,
) -> tuple[str, ReviewReport]:
    report = ReviewReport()
    reviewed = restore_headings_from_sections(text, sections or [], report)
    reviewed = apply_editorial_replacements(reviewed, report)
    reviewed = apply_glossary_bad_aliases(reviewed, glossary_terms or [], report)
    reviewed = apply_gendered_article_fixes(reviewed, glossary_terms or [], report)
    return reviewed, report


def load_sections(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def load_glossary_terms(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    import logging

    state = build_glossary_state(Path(path), None, logging.getLogger(__name__), manual_dir=None)
    return state.manual_terms if state else []
