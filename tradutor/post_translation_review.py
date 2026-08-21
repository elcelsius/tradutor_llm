from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .glossary_utils import build_glossary_state
from .postprocess import final_pt_postprocess
from .quality_checks import run_translation_quality_checks
from .quote_fix import fix_blank_lines_inside_quotes, fix_unbalanced_quotes
from .structure_normalizer import normalize_structure
from .translate import ensure_section_heading


@dataclass
class ReviewReport:
    """Processamento interno auxiliar."""

    heading_fixes: int = 0
    glossary_replacements: dict[str, int] = field(default_factory=dict)
    text_replacements: dict[str, int] = field(default_factory=dict)
    all_caps_name_replacements: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Processamento interno auxiliar."""
        return {
            "heading_fixes": self.heading_fixes,
            "glossary_replacements": self.glossary_replacements,
            "text_replacements": self.text_replacements,
            "all_caps_name_replacements": self.all_caps_name_replacements,
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
        """Processamento interno auxiliar."""
        found = match.group(0)
        if found.isupper():
            return repl.upper()
        if found[:1].isupper():
            return repl[:1].upper() + repl[1:]
        return repl

    return re.subn(pattern, _replace, text, flags=flags)


def _record(counter: dict[str, int], key: str, count: int) -> None:
    """Processamento interno auxiliar."""
    if count:
        counter[key] = counter.get(key, 0) + count


_PREDICATIVE_SUBORDINATE_FIX_RE = re.compile(
    r"(?P<prefix>\b(?:pode|poderia|seria)\s+ser\s+(?:uma|um)\s+"
    r"(?:bênção|pena|sorte|surpresa|vergonha|vantagem|desvantagem|alívio))\s+"
    r"(?P<subject>alguém|ele|ela|um[a]?\s+[A-Za-zÀ-ÿ]+)"
    r"(?P<middle>[^.!?…]{0,100}?)\s+ter\s+sido\s+"
    r"(?P<participle>[A-Za-zÀ-ÿ]+)",
    flags=re.IGNORECASE,
)
_ASSISTIR_PASSIVE_FIX_RE = re.compile(
    r"\b(?P<verb>assistir|assiste|assistiu|assistia|assistiam|assistirá|"
    r"assistirão|assistiriam)\s+os\s+(?=[^.!?…]{0,120}\bserem\b)",
    flags=re.IGNORECASE,
)
_BRAINWASHED_LITERAL_FIX_RE = re.compile(
    r"\b(?P<aux>foi|fui|foram)\s+lav(?:ado|ada|ados|adas)\s+cerebral\b",
    flags=re.IGNORECASE,
)
_BRAINWASHED_BRAIN_LITERAL_FIX_RE = re.compile(
    r"\b(?P<aux>foi|fui|foram)\s+lav(?:ado|ada|ados|adas)\s+o\s+cérebro\b",
    flags=re.IGNORECASE,
)
_BRAINWASHED_INFINITIVE_FIX_RE = re.compile(
    r"\bter\s+sido\s+lav(?:ado|ada|ados|adas)\s+cerebral\b",
    flags=re.IGNORECASE,
)
_FACILITAR_TINGIR_INFINITIVE_FIX_RE = re.compile(
    r"\bfacilita\s+para\s+os\s+outros\s+tingi-la\b",
    flags=re.IGNORECASE,
)
_FACILITAR_TINGIR_SUBJUNCTIVE_FIX_RE = re.compile(
    r"\bfacilita\s+que\s+os\s+outros\s+a\s+tingem\b",
    flags=re.IGNORECASE,
)
_APOLOGY_PROGRESSIVE_FIX_RE = re.compile(
    r"\bEu\s+tô\s+meio\s+que\s+sentindo\s+muito\b",
    flags=re.IGNORECASE,
)
_EYES_DROPPED_FIX_RE = re.compile(
    r"\bOs\s+olhos\s+de\s+(?P<name>[A-ZÀ-Ý][A-Za-zÀ-ÿ]*"
    r"(?:\s+[A-ZÀ-Ý][A-Za-zÀ-ÿ]*)*)\s+(?:caíram|baixaram)\s+para\s+"
    r"as\s+palmas\s+(?:de\s+suas|das)\s+mãos\b",
    flags=re.IGNORECASE,
)
_NOT_FLAWLESS_FIX_RE = re.compile(
    r"\bNão\s+sou\s+(?:falho\s+ou\s+perfeito|perfeito\s+ou\s+impecável)\b",
    flags=re.IGNORECASE,
)
_HARD_TO_TRUST_FIX_RE = re.compile(
    r"\bo\s+que\s+a\s+torna\s+difícil\s+de\s+confiar\b",
    flags=re.IGNORECASE,
)
_FEMININE_MEMBER_FIX_RE = re.compile(
    r"\buma\s+membro\b",
    flags=re.IGNORECASE,
)
_CAPTURE_BRAINWASH_FIX_RE = re.compile(
    r"\bnão\s+o\s+capture\s+e\s+(?:sofreu\s+lavagem\s+cerebral\s+nele|"
    r"lavande\s+o\s+cérebro\s+dele)\b",
    flags=re.IGNORECASE,
)
_HARD_TO_TRUST_RAW_FIX_RE = re.compile(
    r"\bque\s+a\s+gente\s+usa\s+p(?:ra|ara)\s+julgar\s+ela\s+que\s+faz\s+"
    r"ela\s+parecer\s+difícil\s+de\s+confiar\b",
    flags=re.IGNORECASE,
)
_BRIDGE_POSE_FIX_RE = re.compile(
    r"\bna\s+pose\s+de\s+ponte\s+de\s+ginástica\b",
    flags=re.IGNORECASE,
)
_POSSESSIVE_EXPRESSION_FIX_RE = re.compile(
    r"\bPelo\s+seu\s+expressão\b",
    flags=re.IGNORECASE,
)
_MISSING_TREATMENT_NOUN_FIX_RE = re.compile(
    r"\bquando\s+um\s+for\s+desenvolvido\.{1,2}",
    flags=re.IGNORECASE,
)
_CHATTING_AWAY_FIX_RE = re.compile(
    r"\bestavam\s+tendo\s+uma\s+conversa\s+um\s+pouco\s+"
    r"(?:mais\s+)?(?:afastad[oa]s?|distantes?|longe)\b",
    flags=re.IGNORECASE,
)
_CLASSMATES_LOCATION_FIX_RE = re.compile(
    r"\bquantos\s+dos\s+nossos\s+colegas\s+ainda\s+são\s+considerados\s+"
    r"como\s+estando\s+em\s+(?P<place>[A-ZÀ-Ý][A-Za-zÀ-ÿ'’-]*"
    r"(?:\s+[A-ZÀ-Ý][A-Za-zÀ-ÿ'’-]*)*)\s+no\s+momento\b",
    flags=re.IGNORECASE,
)
_BRAINWASH_AND_MANIPULATED_FIX_RE = re.compile(
    r"\btenha\s+sofrido\s+lavagem\s+cerebral\s+e\s+manipulado\b",
    flags=re.IGNORECASE,
)
_SELF_BLAME_FIX_RE = re.compile(
    r"\b(?:É\s+isso\s+que|Isso\s+é\s+o\s+que)\s+a\s+"
    r"(?P<name>[A-ZÀ-Ý][A-Za-zÀ-ÿ]+)\s+culpa\s+pelo\s+seu\s+fracasso\b",
    flags=re.IGNORECASE,
)
_TWO_SEATS_DOWN_FIX_RE = re.compile(
    r"\bNão\s+era\s+ela\s+que\s+estava\s+sentada\s+dois\s+assentos\s+"
    r"ao\s+meu\s+lado\?",
    flags=re.IGNORECASE,
)
_LITTLE_GIRL_NAME_FIX_RE = re.compile(
    r"\bA\s+menina\s+pequena,\s+(?P<name>[A-ZÀ-Ý][A-Za-zÀ-ÿ]+),\s+",
    flags=re.IGNORECASE,
)
_SNORTED_AT_HER_FIX_RE = re.compile(
    r"\bDei\s+um\s+grunhido\s+de\s+deboche\.",
    flags=re.IGNORECASE,
)
_THINKS_TOO_HIGHLY_2C_FIX_RE = re.compile(
    r"\bvocê\s+e\s+os\s+outros\s+membros\s+d(?P<article>[oa])\s+2-C\s+pensam\s+"
    r"(?:muito\s+(?:alto|bem)|demais(?:\s+(?:alto|bem))?)\s+"
    r"(?:de|em)\s+mim\b",
    flags=re.IGNORECASE,
)
_THINKS_TOO_HIGHLY_EVERYONE_FIX_RE = re.compile(
    r"\b(?P<subject>todo\s+mundo|todos)\s+pensa(?:m)?\s+"
    r"(?:muito\s+(?:alto|bem)|demais(?:\s+(?:alto|bem))?)\s+"
    r"(?:de|em)\s+mim\b",
    flags=re.IGNORECASE,
)
_REAL_YOU_FIX_RE = re.compile(
    r"\bAcredito\s+que\s+(?:o\s+verdadeiro|a\s+verdadeira)\s+você\s+"
    r"(?:está|esteja|seja)\s+mais\s+(?:adaptad[oa]|adequad[oa])\s+"
    r"(?:para|a)\s+sobreviver\s+n(?:este|esse)\s+mundo\b",
    flags=re.IGNORECASE,
)
_MUST_ADD_FIX_RE = re.compile(
    r"\bE\s+devo\s+adicionar\s+—",
    flags=re.IGNORECASE,
)
_RECOVER_FROM_FIX_RE = re.compile(
    r"\b(?P<relative>de\s+que|da\s+qual|do\s+qual|das\s+quais|dos\s+quais|que)\s+"
    r"podemos\s+recuperar\b",
    flags=re.IGNORECASE,
)
_RECOVER_CONSCIOUSNESS_REFLEXIVE_FIX_RE = re.compile(
    r"\brecuperar-se\s+da\s+consciência\b",
    flags=re.IGNORECASE,
)
_DUPLICATE_RECOVER_REFLEXIVE_FIX_RE = re.compile(
    r"\bse\s+recuperando-se\b",
    flags=re.IGNORECASE,
)
_RECOVER_SUBJUNCTIVE_SINGULAR_FIX_RE = re.compile(
    r"\bpara\s+que\s+(?P<subject>ele|ela|você|a\s+gente|"
    r"[A-ZÀ-Ý][A-Za-zÀ-ÿ]+)\s+se\s+recuperar\b",
    flags=re.IGNORECASE,
)
_RECOVER_SUBJUNCTIVE_PLURAL_FIX_RE = re.compile(
    r"\bpara\s+que\s+(?P<subject>eles|elas|vocês)\s+se\s+recuperar\b",
    flags=re.IGNORECASE,
)
_BEEN_FAR_FROM_INACTIVE_FIX_RE = re.compile(
    r"\b(?P<subject>ela|ele)\s+tem\s+(?:sido|estado)\s+longe\s+de\s+"
    r"inativ(?P<ending>[oa])\b",
    flags=re.IGNORECASE,
)
_EASY_TO_BRAINWASH_FIX_RE = re.compile(
    r"\bFácil\s+de\s+(?:lavar\s+o\s+cérebro|sofrer\s+lavagem\s+cerebral),\s+"
    r"em\s+outras\s+palavras\.",
    flags=re.IGNORECASE,
)
_ONLY_BRAINWASHED_CAN_TRUST_FIX_RE = re.compile(
    r"\bsó\s+se\s+alguém\s+(?:tivesse\s+sofrido|fosse\s+sofreu)\s+lavagem\s+cerebral\s+"
    r"que\s+conseguiria\s+confiar\b",
    flags=re.IGNORECASE,
)
_HARD_TO_TRUST_SUBORDINATE_FIX_RE = re.compile(
    r"que\s+faz\s+com\s+que\s+(?:ela\s+)?seja\s+difícil\s+(?:de\s+)?confiar"
    r"(?:\s+nela)?",
    flags=re.IGNORECASE,
)
_MISSING_AUXILIARY_BEAT_FIX_RE = re.compile(
    r"\bVocê\s+realmente\s+tentando\s+me\s+vencer\b",
    flags=re.IGNORECASE,
)
_GLAD_CAME_FIX_RE = re.compile(
    r"\b(?P<name>[A-ZÀ-Ý][A-Za-zÀ-ÿ]+)\s+está\s+tão\s+grata\s+por\s+ter\s+vindo\b",
    flags=re.IGNORECASE,
)
_MENTAL_SHOCK_RECOVERY_FIX_RE = re.compile(
    r"\bo\s+choque\s+mental\s+pode\s+levar\s+um\s+tempo\s+para\s+"
    r"(?:ela\s+se\s+recuperar\s+de|superar)\b",
    flags=re.IGNORECASE,
)
_DUPLICATE_JOINED_TOGETHER_FIX_RE = re.compile(
    r"\b(?P<form>(?:ter\s+)?se\s+junt(?:ado|aram))\s+juntos\b",
    flags=re.IGNORECASE,
)
_RIGHT_HAND_POSSESSOR_FIX_RE = re.compile(
    r"\bNa\s+mão\s+direita\s+de\s+(?P<name>[A-ZÀ-Ý][A-Za-zÀ-ÿ]*(?:\s+[A-ZÀ-Ý][A-Za-zÀ-ÿ]*){0,3}),\s+"
    r"ela\s+segurava\s+(?P<object>[^.!?…]{3,180})",
    flags=re.IGNORECASE,
)


def _term_string_list(value: Any) -> list[str]:
    """Processamento interno auxiliar."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _term_variants_for_article_fix(term: dict) -> list[str]:
    """Processamento interno auxiliar."""
    variants: list[str] = []
    for key_field in ("pt", "key"):
        value = str(term.get(key_field, "")).strip()
        if value:
            variants.append(value)
            variants.extend(
                token
                for token in value.split()
                if token[:1].isupper() and len(token) >= 3
            )
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


def apply_high_confidence_grammar_fixes(
    text: str, report: ReviewReport | None = None
) -> str:
    """Apply narrow PT-BR grammar repairs that do not alter literary voice."""
    rpt = report or ReviewReport()

    def fix_predicative_subordinate(match: re.Match[str]) -> str:
        return (
            f"{match.group('prefix')} que {match.group('subject')}"
            f"{match.group('middle')} tenha sido {match.group('participle')}"
        )

    text, count = _PREDICATIVE_SUBORDINATE_FIX_RE.subn(
        fix_predicative_subordinate, text
    )
    _record(rpt.text_replacements, "predicative_subordinate_connector", count)

    def fix_assistir_passive(match: re.Match[str]) -> str:
        return f"{match.group('verb')} aos "

    text, count = _ASSISTIR_PASSIVE_FIX_RE.subn(fix_assistir_passive, text)
    _record(rpt.text_replacements, "assistir_aos_passive", count)

    def fix_brainwashed_literal(match: re.Match[str]) -> str:
        verbs = {"foi": "sofreu", "fui": "sofri", "foram": "sofreram"}
        replacement = f"{verbs[match.group('aux').casefold()]} lavagem cerebral"
        if match.group(0)[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    text, count = _BRAINWASHED_LITERAL_FIX_RE.subn(fix_brainwashed_literal, text)
    _record(rpt.text_replacements, "brainwash_literal_participle", count)

    def fix_brainwashed_brain_literal(match: re.Match[str]) -> str:
        verbs = {"foi": "teve", "fui": "tive", "foram": "tiveram"}
        replacement = f"{verbs[match.group('aux').casefold()]} o cérebro lavado"
        if match.group(0)[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    text, count = _BRAINWASHED_BRAIN_LITERAL_FIX_RE.subn(
        fix_brainwashed_brain_literal, text
    )
    _record(rpt.text_replacements, "brainwash_literal_brain", count)

    text, count = _BRAINWASHED_INFINITIVE_FIX_RE.subn(
        "ter sofrido lavagem cerebral", text
    )
    _record(rpt.text_replacements, "brainwash_literal_infinitive", count)

    text, count = _FACILITAR_TINGIR_INFINITIVE_FIX_RE.subn(
        "facilita que os outros a tinjam", text
    )
    _record(rpt.text_replacements, "facilitar_tingir_subjunctive", count)

    text, count = _FACILITAR_TINGIR_SUBJUNCTIVE_FIX_RE.subn(
        "facilita que os outros a tinjam", text
    )
    _record(rpt.text_replacements, "facilitar_tingir_subjunctive", count)

    text, count = _APOLOGY_PROGRESSIVE_FIX_RE.subn("Eu meio que sinto muito", text)
    _record(rpt.text_replacements, "apology_progressive", count)

    def fix_eyes_dropped(match: re.Match[str]) -> str:
        return f"{match.group('name')} baixou os olhos para as palmas das mãos"

    text, count = _EYES_DROPPED_FIX_RE.subn(fix_eyes_dropped, text)
    _record(rpt.text_replacements, "eyes_dropped", count)

    text, count = _NOT_FLAWLESS_FIX_RE.subn("Não sou impecável nem perfeito", text)
    _record(rpt.text_replacements, "not_flawless", count)

    text, count = _HARD_TO_TRUST_FIX_RE.subn(
        "o que torna difícil confiar nela", text
    )
    _record(rpt.text_replacements, "hard_to_trust", count)

    text, count = _FEMININE_MEMBER_FIX_RE.subn("uma integrante", text)
    _record(rpt.text_replacements, "feminine_member", count)

    text, count = _CAPTURE_BRAINWASH_FIX_RE.subn(
        "não o capture nem o submeta a uma lavagem cerebral", text
    )
    _record(rpt.text_replacements, "capture_brainwash", count)

    text, count = _HARD_TO_TRUST_RAW_FIX_RE.subn(
        "com que a gente a julga que torna difícil confiar nela", text
    )
    _record(rpt.text_replacements, "hard_to_trust", count)

    text, count = _BRIDGE_POSE_FIX_RE.subn("na posição de ponte de ginástica", text)
    _record(rpt.text_replacements, "bridge_pose", count)

    text, count = _POSSESSIVE_EXPRESSION_FIX_RE.subn("Pela sua expressão", text)
    _record(rpt.text_replacements, "possessive_expression", count)

    text, count = _MISSING_TREATMENT_NOUN_FIX_RE.subn(
        "quando um tratamento for desenvolvido.", text
    )
    _record(rpt.text_replacements, "missing_treatment_noun", count)

    text, count = _CHATTING_AWAY_FIX_RE.subn(
        "estavam conversando um pouco mais longe", text
    )
    _record(rpt.text_replacements, "chatting_away", count)

    def fix_classmates_location(match: re.Match[str]) -> str:
        return f"quantos dos nossos colegas ainda devem estar em {match.group('place')}"

    text, count = _CLASSMATES_LOCATION_FIX_RE.subn(fix_classmates_location, text)
    _record(rpt.text_replacements, "classmates_location", count)

    text, count = _BRAINWASH_AND_MANIPULATED_FIX_RE.subn(
        "tenha sofrido lavagem cerebral e sido manipulado", text
    )
    _record(rpt.text_replacements, "brainwash_and_manipulated", count)

    def fix_self_blame(match: re.Match[str]) -> str:
        return f"É por isso que {match.group('name')} se culpa pelo próprio fracasso"

    text, count = _SELF_BLAME_FIX_RE.subn(fix_self_blame, text)
    _record(rpt.text_replacements, "self_blame", count)

    text, count = _TWO_SEATS_DOWN_FIX_RE.subn(
        "Não era ela que estava sentada a dois assentos de distância de mim?", text
    )
    _record(rpt.text_replacements, "two_seats_down", count)

    def fix_little_girl_name(match: re.Match[str]) -> str:
        return f"A pequena {match.group('name')} "

    text, count = _LITTLE_GIRL_NAME_FIX_RE.subn(fix_little_girl_name, text)
    _record(rpt.text_replacements, "little_girl_name", count)

    text, count = _SNORTED_AT_HER_FIX_RE.subn("Bufei para ela.", text)
    _record(rpt.text_replacements, "snorted_at_her", count)

    def fix_2c_thinks_highly(match: re.Match[str]) -> str:
        return (
            "você e os outros membros d"
            f"{match.group('article')} 2-C têm uma opinião elevada demais sobre mim"
        )

    text, count = _THINKS_TOO_HIGHLY_2C_FIX_RE.subn(fix_2c_thinks_highly, text)
    _record(rpt.text_replacements, "thinks_too_highly", count)

    def fix_everyone_thinks_highly(match: re.Match[str]) -> str:
        subject = match.group("subject")
        verb = "tem" if subject.casefold() == "todo mundo" else "têm"
        return f"{subject} {verb} uma opinião elevada demais sobre mim"

    text, count = _THINKS_TOO_HIGHLY_EVERYONE_FIX_RE.subn(
        fix_everyone_thinks_highly, text
    )
    _record(rpt.text_replacements, "thinks_too_highly", count)

    text, count = _REAL_YOU_FIX_RE.subn(
        "Acredito que o seu verdadeiro eu esteja mais apto a sobreviver neste mundo",
        text,
    )
    _record(rpt.text_replacements, "real_you", count)

    text, count = _MUST_ADD_FIX_RE.subn("E devo acrescentar —", text)
    _record(rpt.text_replacements, "must_add", count)

    def fix_recover_from(match: re.Match[str]) -> str:
        relative = match.group("relative")
        if relative.casefold() == "que":
            return "de que podemos nos recuperar"
        return f"{relative} podemos nos recuperar"

    text, count = _RECOVER_FROM_FIX_RE.subn(fix_recover_from, text)
    _record(rpt.text_replacements, "recover_from", count)

    text, count = _RECOVER_CONSCIOUSNESS_REFLEXIVE_FIX_RE.subn(
        "recuperar a consciência", text
    )
    _record(rpt.text_replacements, "recover_consciousness", count)

    text, count = _DUPLICATE_RECOVER_REFLEXIVE_FIX_RE.subn(
        "se recuperando", text
    )
    _record(rpt.text_replacements, "duplicate_recover_reflexive", count)

    text, count = _RECOVER_SUBJUNCTIVE_SINGULAR_FIX_RE.subn(
        r"para que \g<subject> se recupere", text
    )
    _record(rpt.text_replacements, "recover_subjunctive", count)

    text, count = _RECOVER_SUBJUNCTIVE_PLURAL_FIX_RE.subn(
        r"para que \g<subject> se recuperem", text
    )
    _record(rpt.text_replacements, "recover_subjunctive", count)

    def fix_been_far_from_inactive(match: re.Match[str]) -> str:
        return f"{match.group('subject')} tem estado bem ativ{match.group('ending')}"

    text, count = _BEEN_FAR_FROM_INACTIVE_FIX_RE.subn(
        fix_been_far_from_inactive, text
    )
    _record(rpt.text_replacements, "been_far_from_inactive", count)

    text, count = _EASY_TO_BRAINWASH_FIX_RE.subn(
        "Em outras palavras, isso a torna vulnerável à lavagem cerebral.", text
    )
    _record(rpt.text_replacements, "easy_to_brainwash", count)

    def fix_only_brainwashed_can_trust(match: re.Match[str]) -> str:
        replacement = "só alguém que tivesse sofrido lavagem cerebral conseguiria confiar"
        if match.group(0)[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    text, count = _ONLY_BRAINWASHED_CAN_TRUST_FIX_RE.subn(
        fix_only_brainwashed_can_trust, text
    )
    _record(rpt.text_replacements, "only_brainwashed_can_trust", count)

    text, count = _HARD_TO_TRUST_SUBORDINATE_FIX_RE.subn(
        "que torna difícil confiar nela", text
    )
    _record(rpt.text_replacements, "hard_to_trust", count)

    for source, replacement in (
        (
            "que faz com que ela seja difícil de confiar nela",
            "que torna difícil confiar nela",
        ),
        (
            "que faz com que seja difícil de confiar nela",
            "que torna difícil confiar nela",
        ),
        (
            "que faz com que seja difícil confiar nela",
            "que torna difícil confiar nela",
        ),
    ):
        count = text.count(source)
        if count:
            text = text.replace(source, replacement)
        _record(rpt.text_replacements, "hard_to_trust", count)

    text, count = _MISSING_AUXILIARY_BEAT_FIX_RE.subn(
        "Você realmente estava tentando me vencer", text
    )
    _record(rpt.text_replacements, "missing_auxiliary_beat", count)

    def fix_glad_came(match: re.Match[str]) -> str:
        return f"{match.group('name')} está tão contente por ter vindo"

    text, count = _GLAD_CAME_FIX_RE.subn(fix_glad_came, text)
    _record(rpt.text_replacements, "glad_came", count)

    def fix_mental_shock_recovery(match: re.Match[str]) -> str:
        replacement = "o choque mental pode levar algum tempo para ser superado"
        if match.group(0)[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    text, count = _MENTAL_SHOCK_RECOVERY_FIX_RE.subn(
        fix_mental_shock_recovery, text
    )
    _record(rpt.text_replacements, "mental_shock_recovery", count)

    def fix_duplicate_joined_together(match: re.Match[str]) -> str:
        return match.group("form")

    text, count = _DUPLICATE_JOINED_TOGETHER_FIX_RE.subn(
        fix_duplicate_joined_together, text
    )
    _record(rpt.text_replacements, "duplicate_joined_together", count)

    def fix_right_hand_possessor(match: re.Match[str]) -> str:
        return f"{match.group('name')} segurava {match.group('object')} na mão direita"

    text, count = _RIGHT_HAND_POSSESSOR_FIX_RE.subn(fix_right_hand_possessor, text)
    _record(rpt.text_replacements, "right_hand_possessor", count)
    return text


def apply_gendered_article_fixes(
    text: str, glossary_terms: list[dict], report: ReviewReport | None = None
) -> str:
    """Corrige artigos e predicativos incompatíveis com gênero conhecido."""
    rpt = report or ReviewReport()
    for term in glossary_terms:
        gender = str(term.get("gender", "")).strip().casefold()
        category = (
            str(term.get("category") or term.get("type") or term.get("term_type") or "")
            .strip()
            .casefold()
        )
        if (not gender.startswith(("femin", "mascul"))) or (
            "person" not in category
            and "personagem" not in category
            and "criatura" not in category
            and "monstro" not in category
            and "monster" not in category
        ):
            continue
        feminine = gender.startswith("femin")
        for variant in _term_variants_for_article_fix(term):
            escaped = re.escape(variant)
            plural = variant.casefold().endswith("s")
            if feminine:
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
                if plural:
                    article_pairs.extend(
                        [
                            (rf"\bos\s+{escaped}\b", f"as {variant}"),
                            (rf"\bOs\s+{escaped}\b", f"As {variant}"),
                            (rf"\bdos\s+{escaped}\b", f"das {variant}"),
                            (rf"\bDos\s+{escaped}\b", f"Das {variant}"),
                            (rf"\bnos\s+{escaped}\b", f"nas {variant}"),
                            (rf"\bNos\s+{escaped}\b", f"Nas {variant}"),
                            (rf"\baos\s+{escaped}\b", f"às {variant}"),
                            (rf"\bAos\s+{escaped}\b", f"Às {variant}"),
                        ]
                    )
            else:
                article_pairs = [
                    (rf"\ba\s+{escaped}\b", f"o {variant}"),
                    (rf"\bA\s+{escaped}\b", f"O {variant}"),
                    (rf"\bda\s+{escaped}\b", f"do {variant}"),
                    (rf"\bDa\s+{escaped}\b", f"Do {variant}"),
                    (rf"\bna\s+{escaped}\b", f"no {variant}"),
                    (rf"\bNa\s+{escaped}\b", f"No {variant}"),
                    (rf"\bà\s+{escaped}\b", f"ao {variant}"),
                    (rf"\bÀ\s+{escaped}\b", f"Ao {variant}"),
                ]
                if plural:
                    article_pairs.extend(
                        [
                            (rf"\bas\s+{escaped}\b", f"os {variant}"),
                            (rf"\bAs\s+{escaped}\b", f"Os {variant}"),
                            (rf"\bdas\s+{escaped}\b", f"dos {variant}"),
                            (rf"\bDas\s+{escaped}\b", f"Dos {variant}"),
                            (rf"\bnas\s+{escaped}\b", f"nos {variant}"),
                            (rf"\bNas\s+{escaped}\b", f"Nos {variant}"),
                            (rf"\bàs\s+{escaped}\b", f"aos {variant}"),
                            (rf"\bÀs\s+{escaped}\b", f"Aos {variant}"),
                        ]
                    )
            for pattern, repl in article_pairs:
                text, count = re.subn(pattern, repl, text)
                _record(rpt.text_replacements, f"{pattern}->{repl}", count)
            if feminine:
                text, count = re.subn(
                    rf"\b({escaped}) como aliado\b", r"\1 como aliada", text
                )
                _record(
                    rpt.text_replacements,
                    f"{variant} como aliado->{variant} como aliada",
                    count,
                )
            else:
                text, count = re.subn(
                    rf"\b({escaped}) como aliada\b", r"\1 como aliado", text
                )
                _record(
                    rpt.text_replacements,
                    f"{variant} como aliada->{variant} como aliado",
                    count,
                )
    return text


def apply_glossary_bad_aliases(
    text: str, glossary_terms: list[dict], report: ReviewReport | None = None
) -> str:
    """Substitui apenas aliases seguros pelo termo canônico.

    ``contextual_bad_aliases`` não entra aqui: essas formas exigem reparar a
    concordância do trecho completo e são tratadas no repair por LLM.
    """
    rpt = report or ReviewReport()
    insensitive_rules: dict[str, tuple[str, str]] = {}
    sensitive_rules: dict[str, tuple[str, str]] = {}

    def add_rule(alias: str, replacement: str, *, case_sensitive: bool) -> None:
        if not alias or not replacement or alias == replacement:
            return
        rules = sensitive_rules if case_sensitive else insensitive_rules
        key = alias if case_sensitive else alias.casefold()
        # Preserva a prioridade original do glossário quando duas entradas
        # declaram exatamente a mesma forma proibida.
        rules.setdefault(key, (alias, replacement))

    for term in glossary_terms:
        pt = str(term.get("pt", "")).strip()
        if not pt:
            continue
        target_replacements = term.get("target_replacements") or {}
        if isinstance(target_replacements, dict):
            for alias, replacement in target_replacements.items():
                alias_s = str(alias).strip()
                replacement_s = str(replacement).strip()
                add_rule(alias_s, replacement_s, case_sensitive=False)
        bad_aliases = term.get("bad_aliases") or term.get("forbidden_aliases") or []
        if isinstance(bad_aliases, str):
            bad_aliases = [bad_aliases]
        if not isinstance(bad_aliases, list):
            continue
        for alias in bad_aliases:
            alias_s = str(alias).strip()
            # Uma forma proibida que só difere por caixa deve preservar essa
            # diferença: `kyokugen` -> `Kyokugen`, sem reescrever o canônico.
            add_rule(alias_s, pt, case_sensitive=alias_s.casefold() == pt.casefold())

    def apply_rules(
        value: str,
        rules: dict[str, tuple[str, str]],
        *,
        case_sensitive: bool,
    ) -> str:
        if not rules:
            return value
        aliases = sorted(
            (alias for alias, _replacement in rules.values()), key=len, reverse=True
        )
        pattern = re.compile(
            rf"(?<!\w)(?:{'|'.join(re.escape(alias) for alias in aliases)})(?!\w)",
            0 if case_sensitive else re.IGNORECASE,
        )

        def replace(match: re.Match[str]) -> str:
            matched = match.group(0)
            key = matched if case_sensitive else matched.casefold()
            alias, replacement = rules[key]
            _record(rpt.glossary_replacements, f"{alias}->{replacement}", 1)
            return replacement

        return pattern.sub(replace, value)

    # Uma única passada por classe de sensibilidade impede que uma forma curta
    # consuma parte de uma forma proibida mais específica.
    text = apply_rules(text, insensitive_rules, case_sensitive=False)
    return apply_rules(text, sensitive_rules, case_sensitive=True)


def apply_duplicate_canonical_name_fixes(
    text: str, glossary_terms: list[dict], report: ReviewReport | None = None
) -> str:
    """Colapsa duplicatas causadas por expansao indevida de nomes canonicos."""
    rpt = report or ReviewReport()
    for term in glossary_terms:
        pt = str(term.get("pt", "")).strip()
        if not pt or len(pt.split()) < 2:
            continue
        key = str(term.get("key", "")).strip()
        if key.casefold() != pt.casefold():
            continue
        category = (
            str(term.get("category") or term.get("type") or term.get("term_type") or "")
            .strip()
            .casefold()
        )
        if "person" not in category and "personagem" not in category:
            continue
        parts = pt.split()
        first = parts[0]
        last = parts[-1]
        patterns = [
            (rf"(?<!\w){re.escape(first)}\s+{re.escape(pt)}(?!\w)", pt),
            (rf"(?<!\w){re.escape(pt)}\s+{re.escape(last)}(?!\w)", pt),
            (rf"(?<!\w){re.escape(pt)}\s+{re.escape(pt)}(?!\w)", pt),
        ]
        for _ in range(3):
            changed = False
            for pattern, repl in patterns:
                text, count = re.subn(pattern, repl, text, flags=re.IGNORECASE)
                if count:
                    changed = True
                    _record(rpt.text_replacements, f"{pattern}->{repl}", count)
            if not changed:
                break
    return text


def _is_named_entity_term(term: dict) -> bool:
    """Processamento interno auxiliar."""
    category = (
        str(term.get("category") or term.get("type") or term.get("term_type") or "")
        .strip()
        .casefold()
    )
    entity_markers = (
        "person",
        "personagem",
        "criatura",
        "local",
        "organiza",
        "organização",
        "raça",
        "apelido",
    )
    return any(marker in category for marker in entity_markers)


def _can_use_name_parts(term: dict) -> bool:
    """Processamento interno auxiliar."""
    category = (
        str(term.get("category") or term.get("type") or term.get("term_type") or "")
        .strip()
        .casefold()
    )
    return (
        "person" in category
        or "personagem" in category
        or "criatura" in category
        or "apelido" in category
    )


def _looks_like_proper_name(value: str) -> bool:
    """Processamento interno auxiliar."""
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", value)
    if not words:
        return False
    connectors = {"da", "das", "de", "do", "dos", "e"}
    return all(word[:1].isupper() or word.casefold() in connectors for word in words)


def _named_entity_variants(term: dict) -> list[str]:
    """Processamento interno auxiliar."""
    variants: list[str] = []
    for key_field in ("pt", "key", "source_aliases", "aliases", "allowed_target_aliases"):
        value = term.get(key_field)
        values = _term_string_list(value)
        if key_field in {"pt", "key"} and isinstance(value, str):
            values = [value.strip()] if value.strip() else []
        variants.extend(values)

    if _can_use_name_parts(term):
        # Partes isoladas são úteis para nomes pessoais como "Takao Hijiri",
        # mas não para títulos traduzidos como "Irmão Mais Velho do ...":
        # estes fariam palavras comuns em CAPS virarem falsos nomes próprios.
        canonical_pt = str(term.get("pt", "")).strip()
        source_key = str(term.get("key", "")).strip()
        if (
            canonical_pt
            and canonical_pt.casefold() == source_key.casefold()
            and _looks_like_proper_name(canonical_pt)
        ):
            variants.extend(
                part
                for part in canonical_pt.split()
                if len(part) >= 3 and part[:1].isupper()
            )

    unique: dict[str, str] = {}
    for value in variants:
        clean = value.strip()
        if len(clean) < 3 or not _looks_like_proper_name(clean):
            continue
        unique.setdefault(clean.casefold(), clean)
    return sorted(unique.values(), key=len, reverse=True)


def normalize_all_caps_entity_names(
    text: str, glossary_terms: list[dict], report: ReviewReport | None = None
) -> str:
    """Restaura a caixa canônica de entidades que vieram em CAPS do PDF.

    Só toca em nomes explícitos de personagens, criaturas, locais e grupos do
    glossário. Acrônimos e palavras narrativas em maiúsculas ficam intactos.
    """
    rpt = report or ReviewReport()
    for term in glossary_terms:
        if not _is_named_entity_term(term):
            continue
        for variant in _named_entity_variants(term):
            upper = variant.upper()
            if upper == variant:
                continue
            pattern = rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){re.escape(upper)}(?![A-Za-zÀ-ÖØ-öø-ÿ])"
            text, count = re.subn(pattern, variant, text)
            _record(rpt.all_caps_name_replacements, f"{upper}->{variant}", count)
    return text


def _normalized_heading_key(value: str) -> str:
    """Compare headings without Markdown markers, case, or diacritics."""
    plain = value.lstrip("#").strip()
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", plain)
        if not unicodedata.combining(char)
    ).casefold()


def restore_headings_from_sections(
    text: str, sections: list[dict], report: ReviewReport | None = None
) -> str:
    """Reinsere headings que existem no `sections.json` mas sumiram no texto traduzido."""
    rpt = report or ReviewReport()
    if not sections:
        return text
    paragraphs = re.split(r"\n\s*\n", text.strip())
    section_titles = [
        str(sec.get("title", ""))
        for sec in sections
        if str(sec.get("title", "")).strip()
    ]
    section_titles = [title for title in section_titles if title.lower() != "full text"]
    for title in section_titles:
        heading, changed = ensure_section_heading("", title)
        if not heading or not changed:
            continue
        heading_plain = heading.lstrip("#").strip()
        if heading_plain.endswith(":"):
            heading_re = re.compile(
                rf"^#?\s*{re.escape(heading_plain)}(?:\s+\S.*)?\s*$",
                flags=re.IGNORECASE,
            )
        else:
            heading_re = re.compile(
                rf"^#?\s*{re.escape(heading_plain)}\s*$", flags=re.IGNORECASE
            )
        matching_idx: int | None = None
        equivalent_idx: int | None = None
        for idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            first_line = paragraph.strip().splitlines()[0].strip()
            if heading_re.match(first_line):
                matching_idx = idx
                break
            if (
                not heading_plain.endswith(":")
                and _normalized_heading_key(first_line)
                == _normalized_heading_key(heading_plain)
            ):
                equivalent_idx = idx
        if matching_idx is not None:
            continue
        if equivalent_idx is not None:
            lines = paragraphs[equivalent_idx].strip().splitlines()
            if lines and lines[0].strip() != heading:
                lines[0] = heading
                paragraphs[equivalent_idx] = "\n".join(lines)
                rpt.heading_fixes += 1
            continue
        insert_idx = _guess_heading_insert_index(paragraphs, title)
        if insert_idx is None:
            continue
        paragraphs.insert(insert_idx, heading)
        rpt.heading_fixes += 1
    return "\n\n".join(p.strip() for p in paragraphs if p.strip())


def _guess_heading_insert_index(paragraphs: list[str], source_title: str) -> int | None:
    """Processamento interno auxiliar."""
    title = source_title.strip().lower()
    if title.startswith("chapter 1"):
        return _find_first(
            paragraphs,
            (
                r"^Após o (?:Combate Mortal|Deathmatch)\b",
                r"^Depois do (?:Combate Mortal|Deathmatch)\b",
                r"^Após o confronto",
                r"^Depois do confronto",
                r"^Após o Deathmatch",
                r"^Após a ",
                r"^Depois que",
            ),
        )
    if title.startswith("chapter 5"):
        return _find_first(
            paragraphs,
            (
                r"^Minha consciência voltou",
                r"^Minha mente voltou",
                r"^A minha consciência voltou",
            ),
        )
    if title == "prologue":
        return 0
    return None


def _find_first(paragraphs: list[str], patterns: tuple[str, ...]) -> int | None:
    """Processamento interno auxiliar."""
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
    reviewed = apply_high_confidence_grammar_fixes(reviewed, report)
    reviewed = apply_glossary_bad_aliases(reviewed, glossary_terms or [], report)
    reviewed = normalize_all_caps_entity_names(reviewed, glossary_terms or [], report)
    reviewed = apply_duplicate_canonical_name_fixes(
        reviewed, glossary_terms or [], report
    )
    reviewed = apply_gendered_article_fixes(reviewed, glossary_terms or [], report)
    return reviewed, report


def finalize_translation_text(
    text: str,
    *,
    source_text: str = "",
    sections: list[dict] | None = None,
    glossary_terms: list[dict] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Aplica a revisão final não destrutiva e produz um relatório de QA.

    Esta etapa é usada tanto após a tradução quanto após o refine. Ela combina
    normalização estrutural, regras editoriais determinísticas e checagens de
    qualidade; não chama LLM e não muda o sentido do texto.
    """
    terms = glossary_terms or []
    normalized = normalize_structure(final_pt_postprocess(text))
    reviewed, editorial = review_translation_text(
        normalized, sections=sections, glossary_terms=terms
    )
    reviewed = normalize_structure(final_pt_postprocess(reviewed))
    reviewed, quote_balance_fixed = fix_unbalanced_quotes(reviewed)
    if quote_balance_fixed:
        # ``fix_unbalanced_quotes`` pode inserir o fechamento antes da próxima
        # abertura. Quando essa abertura inicia um novo parágrafo, move o
        # fechamento para o fim da fala anterior antes da limpeza final.
        reviewed = re.sub(
            r"(?m)([^\n])\n(?:[ \t]*\n)+[ \t]*”(?=“)", r"\1”\n\n", reviewed
        )
        reviewed = normalize_structure(final_pt_postprocess(reviewed))
    reviewed, quote_blank_lines_fixed = fix_blank_lines_inside_quotes(reviewed)
    if quote_blank_lines_fixed:
        reviewed = normalize_structure(final_pt_postprocess(reviewed))
    quality = run_translation_quality_checks(source_text, reviewed, terms)
    return reviewed, {
        "editorial": editorial.to_dict(),
        "quote_blank_lines_fixed": quote_blank_lines_fixed,
        "quote_balance_fixed": quote_balance_fixed,
        "quality": quality,
    }


def load_sections(path: str | Path | None) -> list[dict]:
    """Processamento interno auxiliar."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def load_glossary_terms(path: str | Path | None) -> list[dict]:
    """Processamento interno auxiliar."""
    if not path:
        return []
    import logging

    state = build_glossary_state(
        Path(path), None, logging.getLogger(__name__), manual_dir=None
    )
    return state.manual_terms if state else []
