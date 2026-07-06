from __future__ import annotations

import re
from collections import Counter
from typing import Any

GlossaryEntry = dict[str, Any]

TRANSLATION_MARKER_RE = re.compile(
    r"###\s*(?:TEXTO_TRADUZ[A-Z_]*|TEXTO_REFINADO_(?:INICIO|FIM))",
    re.IGNORECASE,
)
ENGLISH_META_RE = re.compile(
    r"\b(The narrative|Key characters|story highlights|training grounds|Chapter\s+\d+)\b",
    re.IGNORECASE,
)
DEFAULT_SOURCE_LEAKS = (
    "S-class",
    "Dragonslayer",
    "Four Holy Elders",
    "Sabre-Toothed Tigers",
    "Golden-Eyed Monsters",
    "Land of the Golden-Eyed Monsters",
    "Monster Slayer King",
    "Monster Slayer Knights",
    "Phew",
    "Geez",
    "Huh",
    "Ugh",
    "ain't",
    "ain’t",
)

FEMININE_MASCULINE_MARKERS = (
    "atento",
    "envergonhado",
    "culpado",
    "calado",
    "quieto",
    "sozinho",
    "preparado",
    "confuso",
    "surpreso",
    "irritado",
    "exausto",
    "sentado",
    "parado",
    "ferido",
    "machucado",
    "louco",
    "alto",
)
MASCULINE_FEMININE_MARKERS = (
    "atenta",
    "envergonhada",
    "culpada",
    "calada",
    "quieta",
    "sozinha",
    "preparada",
    "confusa",
    "surpresa",
    "irritada",
    "exausta",
    "sentada",
    "parada",
    "ferida",
    "machucada",
    "louca",
)
MIXED_GENDER_PATTERNS = (
    (r"\bcuidadosa\b.{0,50}\batento\b", "cuidadosa/atento"),
    (r"\batenta\b.{0,50}\bcuidadoso\b", "atenta/cuidadoso"),
    (r"\benvergonhada\b.{0,50}\bculpado\b", "envergonhada/culpado"),
    (r"\benvergonhado\b.{0,50}\bculpada\b", "envergonhado/culpada"),
    (r"\bpreparada\b.{0,50}\bpronto\b", "preparada/pronto"),
    (r"\bpreparado\b.{0,50}\bpronta\b", "preparado/pronta"),
)
MASCULINE_NOUN_CONTEXT_RE = re.compile(r"\b(?:olhar|rosto|semblante|sorriso|tom)\b", re.IGNORECASE)


def _term_variants(term: GlossaryEntry) -> list[str]:
    variants = [str(term.get("key", "")).strip()]
    aliases = term.get("source_aliases") or term.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    if isinstance(aliases, list):
        variants.extend(str(alias).strip() for alias in aliases if str(alias).strip())
    return list(dict.fromkeys(v for v in variants if v))


def _bad_aliases(term: GlossaryEntry) -> list[str]:
    aliases = term.get("bad_aliases") or term.get("forbidden_aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list):
        return []
    return [str(alias).strip() for alias in aliases if str(alias).strip()]


def _allowed_target_aliases(term: GlossaryEntry) -> list[str]:
    aliases = term.get("allowed_target_aliases") or term.get("target_aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list):
        return []
    return [str(alias).strip() for alias in aliases if str(alias).strip()]


def _contains(text: str, needle: str) -> bool:
    if not text or not needle:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)", re.IGNORECASE)
    return bool(pattern.search(text))


def _snippet(text: str, needle: str, radius: int = 70) -> str:
    match = re.search(re.escape(needle), text, flags=re.IGNORECASE)
    if not match:
        return ""
    start = max(match.start() - radius, 0)
    end = min(match.end() + radius, len(text))
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _add_issue(
    issues: list[dict[str, str]],
    issue_type: str,
    message: str,
    *,
    term: str = "",
    found: str = "",
    snippet: str = "",
) -> None:
    issues.append(
        {
            "type": issue_type,
            "term": term,
            "found": found,
            "message": message,
            "snippet": snippet,
        }
    )


def _is_character(term: GlossaryEntry) -> bool:
    category = str(term.get("category", "")).strip().lower()
    term_type = str(term.get("type") or term.get("term_type") or "").strip().lower()
    return category == "personagem" or term_type == "personagem"


def _check_glossary(
    source_text: str,
    translated_text: str,
    glossary_terms: list[GlossaryEntry],
    issues: list[dict[str, str]],
) -> None:
    for term in glossary_terms:
        key = str(term.get("key", "")).strip()
        pt = str(term.get("pt", "")).strip()
        if not key or not pt:
            continue

        variants = _term_variants(term)
        allowed_target_aliases = {alias.casefold() for alias in _allowed_target_aliases(term)}
        source_has_term = any(_contains(source_text, variant) for variant in variants)
        target_has_pt = _contains(translated_text, pt)

        for bad_alias in _bad_aliases(term):
            if _contains(translated_text, bad_alias):
                _add_issue(
                    issues,
                    "bad_alias_in_target",
                    "Alias conhecido como erro apareceu na tradução.",
                    term=key,
                    found=bad_alias,
                    snippet=_snippet(translated_text, bad_alias),
                )

        if key != pt:
            for variant in variants:
                if variant == pt:
                    continue
                if _contains(pt, variant):
                    continue
                if variant.casefold() in allowed_target_aliases:
                    continue
                if _contains(translated_text, variant):
                    _add_issue(
                        issues,
                        "source_term_in_target",
                        "Termo do original ou variante não canônica vazou na tradução.",
                        term=key,
                        found=variant,
                        snippet=_snippet(translated_text, variant),
                    )

        if source_has_term and key != pt and not target_has_pt:
            _add_issue(
                issues,
                "missing_canonical_term",
                "O original usa o termo, mas a tradução canônica não apareceu.",
                term=key,
                found="",
                snippet="",
            )


def _check_default_source_leaks(translated_text: str, issues: list[dict[str, str]]) -> None:
    for phrase in DEFAULT_SOURCE_LEAKS:
        if _contains(translated_text, phrase):
            _add_issue(
                issues,
                "residual_english",
                "Expressão inglesa conhecida apareceu no texto traduzido.",
                term=phrase,
                found=phrase,
                snippet=_snippet(translated_text, phrase),
            )
    for match in ENGLISH_META_RE.finditer(translated_text):
        found = match.group(0)
        _add_issue(
            issues,
            "english_meta_text",
            "Trecho parece metacomentário ou resumo em inglês, não tradução literária.",
            term="",
            found=found,
            snippet=_snippet(translated_text, found),
        )


def _check_gender(translated_text: str, glossary_terms: list[GlossaryEntry], issues: list[dict[str, str]]) -> None:
    sentences = re.split(r"(?<=[.!?…])\s+|\n+", translated_text)
    for sentence in sentences:
        if not sentence.strip():
            continue
        for pattern, label in MIXED_GENDER_PATTERNS:
            if re.search(pattern, sentence, flags=re.IGNORECASE | re.DOTALL):
                _add_issue(
                    issues,
                    "possible_gender_mismatch",
                    "Possível mistura de feminino e masculino na mesma construção.",
                    found=label,
                    snippet=re.sub(r"\s+", " ", sentence).strip()[:220],
                )
    for term in glossary_terms:
        if not _is_character(term):
            continue
        gender = str(term.get("gender", "")).strip().lower()
        if gender not in {"feminino", "masculino"}:
            continue
        key = str(term.get("key", "")).strip()
        pt = str(term.get("pt", key)).strip()
        names = [pt, key]
        if " " in pt:
            names.extend(part for part in pt.split() if len(part) > 3)
        markers = FEMININE_MASCULINE_MARKERS if gender == "feminino" else MASCULINE_FEMININE_MARKERS
        marker_re = re.compile(rf"\b({'|'.join(re.escape(m) for m in markers)})\b", re.IGNORECASE)
        for sentence in sentences:
            if not sentence.strip():
                continue
            if not any(_contains(sentence, name) for name in names):
                continue
            match = marker_re.search(sentence)
            if not match:
                continue
            before_expression = sentence[max(0, match.start() - 12) : match.start()]
            if re.search(r"\bde\s+$", before_expression, flags=re.IGNORECASE):
                continue
            if gender == "feminino":
                before = sentence[max(0, match.start() - 40) : match.start()]
                if MASCULINE_NOUN_CONTEXT_RE.search(before):
                    continue
            _add_issue(
                issues,
                "possible_gender_mismatch",
                "Possível discordância de gênero perto de personagem com gênero conhecido.",
                term=key,
                found=match.group(1),
                snippet=re.sub(r"\s+", " ", sentence).strip()[:220],
            )


def _check_structure(translated_text: str, issues: list[dict[str, str]]) -> None:
    marker = TRANSLATION_MARKER_RE.search(translated_text)
    if marker:
        _add_issue(
            issues,
            "residual_translation_marker",
            "Marcador interno de tradução permaneceu na saída.",
            found=marker.group(0),
            snippet=_snippet(translated_text, marker.group(0)),
        )

    if translated_text.count('"') % 2:
        _add_issue(
            issues,
            "unbalanced_quotes",
            "Quantidade ímpar de aspas retas pode indicar fala quebrada.",
            found='"',
            snippet="",
        )
    if translated_text.count("“") != translated_text.count("”"):
        _add_issue(
            issues,
            "unbalanced_quotes",
            "Aspas curvas de abertura/fechamento estão desbalanceadas.",
            found="“/”",
            snippet="",
        )


def run_translation_quality_checks(
    source_text: str,
    translated_text: str,
    glossary_terms: list[GlossaryEntry] | None = None,
    *,
    max_issues: int = 80,
) -> dict[str, Any]:
    """Executa checagens determinísticas de qualidade para comparar saídas de benchmark."""
    terms = glossary_terms or []
    issues: list[dict[str, str]] = []
    _check_glossary(source_text, translated_text, terms, issues)
    _check_default_source_leaks(translated_text, issues)
    _check_gender(translated_text, terms, issues)
    _check_structure(translated_text, issues)

    penalties = {
        "bad_alias_in_target": 10,
        "source_term_in_target": 8,
        "missing_canonical_term": 6,
        "residual_english": 5,
        "english_meta_text": 8,
        "possible_gender_mismatch": 3,
        "residual_translation_marker": 10,
        "unbalanced_quotes": 4,
    }
    score = 100
    for issue in issues:
        score -= penalties.get(issue["type"], 2)
    score = max(score, 0)

    by_type = Counter(issue["type"] for issue in issues)
    return {
        "score": score,
        "issue_count": len(issues),
        "issues_by_type": dict(sorted(by_type.items())),
        "issues": issues[:max_issues],
        "truncated": len(issues) > max_issues,
    }


def format_quality_cell(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    by_type = report.get("issues_by_type") or {}
    if not by_type:
        return f"{report.get('score', 0)}/100"
    top = ", ".join(f"{key}:{value}" for key, value in sorted(by_type.items())[:3])
    return f"{report.get('score', 0)}/100 ({top})"
