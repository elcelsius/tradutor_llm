"""Conservative source-aware review for translated chunks.

The translation model is responsible for the prose.  This stage is deliberately
smaller: it asks a second model for a short list of exact substitutions, then
applies only the suggestions that preserve structure and objective QA.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .anti_hallucination import detect_entity_mutation
from .cache_utils import cache_exists, chunk_hash, detect_model_collapse, load_cache, save_cache
from .language_guardrails import detect_residual_english
from .llm_backend import LLMBackend
from .qa import count_quotes
from .quality_checks import run_translation_quality_checks


BILINGUAL_REVIEW_PIPELINE_VERSION = "17"
DEFAULT_MAX_CHANGES = 6
_HONORIFIC_SUFFIXES = r"san|kun|chan|sama|dono|sensei"
_PROPER_NAME_ARTICLE_STYLE_RE = re.compile(
    r"^(?P<prefix>.*\b)(?P<article>(?i:da|do|das|dos|a|o|as|os))\s+"
    r"(?P<name>[A-ZÀ-Ý][A-Za-zÀ-ÿ]*(?:[ -][A-ZÀ-Ý][A-Za-zÀ-ÿ]*)*"
    rf"(?:-(?:{_HONORIFIC_SUFFIXES}))?)"
    r"(?P<suffix>[.,;:!?…]?)$"
)
_INLINE_PROPER_NAME_ARTICLE_STYLE_RE = re.compile(
    r"(?<![A-Za-zÀ-ÿ])(?P<article>(?i:da|do|das|dos|a|o|as|os))\s+"
    r"(?P<name>[A-ZÀ-Ý][A-Za-zÀ-ÿ]*(?:[ -][A-ZÀ-Ý][A-Za-zÀ-ÿ]*)*"
    rf"(?:-(?:{_HONORIFIC_SUFFIXES}))?)"
)
_PT_PERIPHRASTIC_PERFECT_RE = re.compile(
    r"\b(?:tenho|tem|temos|têm|tinha|tinham|teria|teriam|"
    r"terei|teremos|terá|terão)\s+(?:se\s+)?"
    r"[A-Za-zÀ-ÿ]+(?:ado|ada|ados|adas|ido|ida|idos|idas)\b",
    flags=re.IGNORECASE,
)
_DIALOGUE_TREATMENT_CLITIC_RE = re.compile(
    r"\b[A-Za-zÀ-ÿ]+-(?P<clitic>te|se|vos|nos)\b",
    flags=re.IGNORECASE,
)
_SHORT_PHRASE_STOPWORDS = {"a", "as", "o", "os", "de", "da", "das", "do", "dos"}
_POSSESSIVE_LOCATIVE_FOCUS_RE = re.compile(
    r"\b(?:na|no)\s+(?:mão|braço|ombro)\s+(?:direita|esquerda)\s+de\s+"
    r"[A-ZÀ-Ý][A-Za-zÀ-ÿ]*(?:[ -][A-ZÀ-Ý][A-Za-zÀ-ÿ]*){0,3},\s+"
    r"(?:ela|ele)\s+[A-Za-zÀ-ÿ]+",
    flags=re.IGNORECASE,
)
_MISSING_SUBORDINATE_CONNECTOR_FOCUS_RE = re.compile(
    r"\b(?:pode|poderia|seria)\s+ser\s+(?:uma|um)\s+"
    r"(?:bênção|pena|sorte|surpresa|vergonha|vantagem|desvantagem|alívio)\s+"
    r"(?:alguém|ele|ela|um[a]?\s+[A-Za-zÀ-ÿ]+)[^.!?…]{0,100}\bter\s+sido\b",
    flags=re.IGNORECASE,
)
_CAPITALIZED_TOKEN_RE = re.compile(
    r"(?<![A-Za-zÀ-ÿ])(?P<token>[A-ZÀ-Ý][A-Za-zÀ-ÿ]{2,})(?![A-Za-zÀ-ÿ])"
)
_GENDERED_PARTICIPLE_RE = re.compile(
    r"\b(?P<stem>[A-Za-zÀ-ÿ]+?)(?P<ending>ada|ado|ida|ido)\b",
    flags=re.IGNORECASE,
)
_BRAINWASH_CALQUE_RE = re.compile(
    r"\b(?:lav(?:ado|ada|ados|adas)|lavande)\s+(?:o\s+)?cérebro\b|"
    r"\blav(?:ado|ada|ados|adas)\s+cerebral\b",
    flags=re.IGNORECASE,
)
_BRAINWASH_CANONICAL_RE = re.compile(
    r"\b(?:lavagem\s+cerebral|cérebro\s+lavado)\b",
    flags=re.IGNORECASE,
)
_LITERAL_MADE_MOVE_RE = re.compile(
    r"\bf(?:ez|izeram)\s+(?:o|seu|sua)\s+movimento\b",
    flags=re.IGNORECASE,
)
_UNGRAMMATICAL_TINGIR_SUBJUNCTIVE_RE = re.compile(
    r"\bfacilita\s+que\s+(?:os|as)\s+outros\s+[oa]\s+tingem\b",
    flags=re.IGNORECASE,
)
_COLLECTIVE_SINGULAR_PREDICATE_RE = re.compile(
    r"\bA\s+maioria\b[^.!?…]{0,120}\b(?:é|foi|era|está|estava|será)\b",
    flags=re.IGNORECASE,
)
_COLLECTIVE_PLURAL_PREDICATE_RE = re.compile(
    r"\bA\s+maioria\b[^.!?…]{0,120}\b(?:são|foram|eram|estão|estavam|serão)\b",
    flags=re.IGNORECASE,
)
_RECOVER_FROM_CALQUE_RE = re.compile(
    r"\b(?:de\s+que|da\s+qual|do\s+qual|das\s+quais|dos\s+quais)\s+"
    r"podemos\s+recuperar\b",
    flags=re.IGNORECASE,
)
_RECOVER_CONSCIOUSNESS_REFLEXIVE_RE = re.compile(
    r"\brecuperar-se\s+d[ao]s?\s+consciênc(?:ia|ias)\b",
    flags=re.IGNORECASE,
)
_DUPLICATE_RECOVER_REFLEXIVE_RE = re.compile(
    r"\bse\s+recuperando-se\b",
    flags=re.IGNORECASE,
)
_RECOVER_SUBJUNCTIVE_AFTER_PARA_QUE_RE = re.compile(
    r"\bpara\s+que\s+(?:(?:ele|ela|eles|elas|você|vocês|a\s+gente|"
    r"[A-ZÀ-Ý][A-Za-zÀ-ÿ]+)\s+)?se\s+recuperar\b",
    flags=re.IGNORECASE,
)
_LEMBREI_PRONOMINAL_RE = re.compile(r"\blembrei-me\b", flags=re.IGNORECASE)
_LEMBREI_WITH_DE_RE = re.compile(
    r"\blembrei\s+d(?:a|as|o|os)\s+", flags=re.IGNORECASE
)
_LEMBREI_DIRECT_OBJECT_RE = re.compile(
    r"\blembrei\s+(?:a|as|o|os)\s+", flags=re.IGNORECASE
)
_COLLOQUIAL_DIRECT_OBJECT_RE = re.compile(
    r"\b(?P<verb>[A-Za-zÀ-ÿ]+)\s+(?P<object>ele|ela|eles|elas)\b",
    flags=re.IGNORECASE,
)
_FORMAL_PREVERBAL_CLITIC_RE = re.compile(
    r"\b(?P<clitic>o|a|os|as)\s+(?P<verb>[A-Za-zÀ-ÿ]+)\b",
    flags=re.IGNORECASE,
)
_COLLOQUIAL_AGRADECER_VOCE_RE = re.compile(
    r"\bagradecer\s+voc(?:ê|ês)\b", flags=re.IGNORECASE
)
_FORMAL_AGRADECER_CLITIC_RE = re.compile(
    r"\bagradecer-(?:lhe|lhes)\b", flags=re.IGNORECASE
)
_COLLOQUIAL_DEIXAR_EU_RE = re.compile(
    r"\bdeixar\s+eu\s+[A-Za-zÀ-ÿ]+\b", flags=re.IGNORECASE
)
_FORMAL_DEIXAR_QUE_EU_RE = re.compile(
    r"\bdeixar\s+que\s+eu\s+[A-Za-zÀ-ÿ]+\b", flags=re.IGNORECASE
)
_GENERIC_GLOSSARY_TITLE_TOKENS = frozenset(
    {
        "aldeia",
        "castle",
        "cidade",
        "city",
        "continente",
        "country",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "distrito",
        "empire",
        "estado",
        "forest",
        "fortaleza",
        "ilha",
        "império",
        "island",
        "kingdom",
        "local",
        "lord",
        "mar",
        "montanha",
        "mountain",
        "ordem",
        "país",
        "reino",
        "região",
        "ruína",
        "ruínas",
        "senhor",
        "senhora",
        "state",
        "terra",
        "torre",
        "vale",
        "vila",
    }
)
_REVIEW_OVERLAP_STOPWORDS = frozenset(
    {
        "a",
        "ao",
        "aos",
        "as",
        "com",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "ela",
        "ele",
        "em",
        "essa",
        "esse",
        "esta",
        "este",
        "eu",
        "foi",
        "o",
        "os",
        "para",
        "por",
        "que",
        "se",
        "ser",
        "sua",
        "seu",
        "um",
        "uma",
    }
)
_DANGLING_PREPOSITION_RE = re.compile(
    r"\b(?:a|com|de|em|entre|para|por|sobre)\s*[.!?…](?:[\"”’']*)\s*(?:$|\n)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_INVALID_BRAINWASH_AUXILIARY_RE = re.compile(
    r"\b(?:foi|fosse|era|seria|seja|será)\s+sofreu\s+lavagem\s+cerebral\b",
    flags=re.IGNORECASE,
)
_DUPLICATE_JOINED_TOGETHER_RE = re.compile(
    r"\b(?:ter\s+)?se\s+junt(?:ado|aram)\s+juntos\b",
    flags=re.IGNORECASE,
)


@dataclass
class BilingualReviewResult:
    """Outcome of one conservative source-aware review call."""

    text: str
    attempted: bool = False
    changed: bool = False
    used_cache: bool = False
    llm_attempts: int = 0
    suggested_changes: int = 0
    applied_changes: list[dict[str, str]] = field(default_factory=list)
    rejected_changes: list[dict[str, str]] = field(default_factory=list)
    failure_reason: str = ""
    raw_output: str = ""
    elapsed_seconds: float = 0.0
    focus_issues: list[str] = field(default_factory=list)


def detect_bilingual_review_focuses(translated_text: str) -> list[str]:
    """Return short, generic PT-BR patterns that merit source-aware review.

    These are prompts for inspection, not deterministic transformations. The
    reviewer still needs the English source to confirm that a correction is
    appropriate.
    """
    focuses: list[str] = []
    for pattern in (
        _POSSESSIVE_LOCATIVE_FOCUS_RE,
        _MISSING_SUBORDINATE_CONNECTOR_FOCUS_RE,
    ):
        for match in pattern.finditer(translated_text):
            excerpt = re.sub(r"\s+", " ", match.group(0)).strip()
            if excerpt and excerpt not in focuses:
                focuses.append(excerpt)
            if len(focuses) >= 3:
                return focuses
    return focuses


def build_bilingual_review_prompt(
    *,
    source_text: str,
    translated_text: str,
    glossary_text: str | None = None,
    focus_issues: list[str] | None = None,
    max_changes: int = DEFAULT_MAX_CHANGES,
) -> str:
    """Build a minimal-diff bilingual review prompt."""
    glossary_block = ""
    if glossary_text:
        glossary_block = (
            "GLOSSARIO RELEVANTE (nao altere formas canonicas):\n"
            f"{glossary_text}\n\n"
        )
    focus_block = ""
    if focus_issues:
        focus_lines = "\n".join(f"- {issue}" for issue in focus_issues[:3])
        focus_block = (
            "FOCOS DE INSPECAO (sao alertas, nao correcoes obrigatorias):\n"
            f"{focus_lines}\n"
            "Verifique cada foco contra o ORIGINAL antes de propor qualquer troca.\n\n"
        )
    return f"""
Voce e um REVISOR BILINGUE CONSERVADOR de light novels EN->PT-BR.

Compare o ORIGINAL e a TRADUCAO ATUAL. Identifique no maximo {max_changes} correcoes
de ALTA CONFIANCA: erro claro de sentido, gramatica, regencia, concordancia ou
calque artificial. Nao reescreva por preferencia pessoal e nao altere giria
  intencional, nomes proprios, honorificos ou formas canonicas do glossario.
  Nao trate a alternancia opcional de artigo antes de nome proprio como erro
  (por exemplo, "da Hijiri" versus "de Hijiri", ou "a Vicius" versus
  "Vicius").
  Nao troque uma unica palavra por mero sinonimo ou por registro mais formal
  (por exemplo, "nao tem" por "nao possui").
  Nao espelhe mecanicamente o tempo/aspecto verbal do ingles: o present
  perfect de uma acao acabada normalmente continua natural em preterito
  perfeito no PT-BR. Em especial, nao troque um preterito simples por
  "tem/tenho/têm + participio" apenas porque o original usa "has/have".
  Nao regularize a pessoa de tratamento de uma fala. Formas como
  "ajoelha-te", "apressa-te", "vós" e construcoes equivalentes podem ser
  parte da voz do personagem, mesmo se o mesmo dialogo tambem usar "você".
  Ao corrigir um sintagma nominal, inclua artigo e adjetivos afetados; nao
  reorganize somente duas palavras se a concordancia estiver fora do trecho.
  Nunca troque, complete ou adivinhe um nome proprio. Se houver duvida sobre
  uma forma de nome, mantenha a traducao atual. Cada item precisa de um motivo
  objetivo; nao envie sugestoes sem motivo. Nao use construcoes como
  "lavada cerebral" ou "lavada o cerebro" para "brainwashed". A forma
  permitida e canonica para esse sentido e "sofreu lavagem cerebral"; nao a
  substitua pelo sentido mais fraco de "foi manipulada".
  Nao altere uma concordancia aceitavel de coletivo, como "a maioria ... e",
  para plural apenas por preferencia. Tambem nao elimine artigos opcionais
  antes de nomes proprios quando eles aparecem dentro de uma frase.
  Nao troque um futuro periferico natural de fala, como "vai depender", por
  futuro simples apenas para tornar o registro mais formal.
  Nao acrescente ou remova o pronome em "lembrei"/"lembrei-me" apenas por
  preferencia de registro. Tambem nao troque "lembrei de/das ..." por
  "lembrei ..." apenas por uma preferencia de regencia: a forma atual e
  natural em PT-BR e deve ser mantida. Para "recover from", nao use
  "recuperar de"; a forma correta e "recuperar-se de". Para o eliptico
  "easy to brainwash", nao escreva "facil de sofrer lavagem cerebral".
  Essa regra vale somente para "recover from": "recover consciousness" e
  "recuperar a consciencia", nunca "recuperar-se da consciencia". Para
  recuperacao de saude sem complemento, use "se recuperar"; apos "para que",
  preserve o subjuntivo (por exemplo, "para que ela se recupere") e nunca
  duplique o pronome em "se recuperando-se".
  Preserve construcoes coloquiais naturais em dialogos, como objeto tonico
  apos o verbo ("estudamos ele", "provocaram ela"), "agradecer voce" e
  "deixar eu + verbo". Nao as troque por cliticos preverbais, "agradecer-lhe"
  ou "deixar que eu ..." apenas para impor registro mais formal.
  Uma sugestao nunca pode deixar uma preposicao sem complemento, como
  "recuperar de.". Preserve ao menos uma palavra de conteudo da traducao
  original: se a troca nao compartilha nenhuma ancora lexical, ela provavelmente
  nao e uma correcao minima e deve ser descartada.

  Antes de responder, faca uma passada objetiva procurando: sujeito e verbo
  compativeis, genero e numero de pronomes, referente correto de cada pronome,
  preposicoes/regencia, conectivos obrigatorios (como o "que" de uma oracao
  encaixada) e locativos possessivos com sujeito ambiguo. Por exemplo, se o
  original confirma que X e o sujeito de "In X's right hand, she held...",
  a forma PT deve deixar X como sujeito, sem manter um "ela" ambiguo. Corrija
  somente quando o ORIGINAL confirmar o problema.

Para cada correcao, `original` deve ser uma substring EXATA da traducao atual e
`substituicao` deve ser a correcao minima. Preserve todos os paragrafos, aspas,
eventos e a voz dos personagens. Nao resuma, nao acrescente informacao e nao
explique a resposta fora do JSON. Se nao houver correcao de alta confianca,
retorne uma lista vazia.

Responda exclusivamente JSON valido neste formato:
{{"correcoes":[{{"original":"...","substituicao":"...","motivo":"..."}}]}}

 {glossary_block}{focus_block}ORIGINAL EM INGLES:
\"\"\"{source_text}\"\"\"

TRADUCAO ATUAL:
\"\"\"{translated_text}\"\"\""""


def bilingual_review_prompt_fingerprint() -> str:
    """Return a stable cache fingerprint for the review prompt contract."""
    prompt = build_bilingual_review_prompt(
        source_text="{source}",
        translated_text="{translated}",
        glossary_text="{glossary}",
        focus_issues=["{focus}"],
    )
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _json_object_from_response(raw: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response, including fenced JSON."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            payload, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def parse_bilingual_review_output(raw: str, *, max_changes: int) -> list[dict[str, str]]:
    """Parse and normalize the model's list of exact substitutions."""
    payload = _json_object_from_response(raw)
    if payload is None:
        raise ValueError("review_invalid_json")
    corrections = payload.get("correcoes")
    if not isinstance(corrections, list):
        raise ValueError("review_missing_correcoes")

    parsed: list[dict[str, str]] = []
    for item in corrections[:max_changes]:
        if not isinstance(item, dict):
            continue
        original = str(item.get("original") or "").strip()
        replacement = str(
            item.get("substituicao") or item.get("substituição") or ""
        ).strip()
        reason = str(item.get("motivo") or "").strip()
        if original and replacement:
            parsed.append(
                {
                    "original": original,
                    "substituicao": replacement,
                    "motivo": reason,
                }
            )
    return parsed


def _paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()])


def _remove_optional_proper_name_article(value: str) -> str | None:
    """Normalize a stylistic article before a proper name, when present."""
    match = _PROPER_NAME_ARTICLE_STYLE_RE.fullmatch(value.strip())
    if not match:
        return None
    article = match.group("article").casefold()
    connector = "de " if article in {"da", "do", "das", "dos"} else ""
    return f"{match.group('prefix')}{connector}{match.group('name')}{match.group('suffix')}"


def _normalize_optional_proper_name_articles(value: str) -> str:
    """Remove only optional articles before a capitalized name/honorific."""
    full_normalized = _remove_optional_proper_name_article(value)
    candidate = full_normalized if full_normalized is not None else value

    def replace(match: re.Match[str]) -> str:
        article = match.group("article").casefold()
        connector = "de " if article in {"da", "do", "das", "dos"} else ""
        return f"{connector}{match.group('name')}"

    return _INLINE_PROPER_NAME_ARTICLE_STYLE_RE.sub(replace, candidate)


def _is_optional_proper_name_article_change(original: str, replacement: str) -> bool:
    """Reject a stylistic article normalization before a proper name."""
    normalize = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
    return normalize(_normalize_optional_proper_name_articles(original)) == normalize(
        _normalize_optional_proper_name_articles(replacement)
    )


def _protected_glossary_forms(terms: list[dict] | None) -> set[str]:
    """Return locked/proper-name glossary variants that a reviewer may not alter."""
    forms: set[str] = set()
    for term in terms or []:
        category = str(term.get("category") or term.get("type") or "").casefold()
        protected = bool(term.get("locked") or term.get("enforce")) or category in {
            "personagem",
            "local",
            "criatura",
        }
        if not protected:
            continue
        values = [term.get("key"), term.get("pt")]
        for field in (
            "source_aliases",
            "aliases",
            "allowed_target_aliases",
        ):
            raw = term.get(field) or []
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, list):
                values.extend(raw)
        for value in values:
            form = str(value or "").strip()
            if len(form) >= 3 and any(char.isupper() for char in form):
                forms.add(form)
                if category in {"personagem", "local", "criatura"}:
                    forms.update(
                        token
                        for token in (
                            match.group("token")
                            for match in _CAPITALIZED_TOKEN_RE.finditer(form)
                        )
                        if token.casefold() not in _GENERIC_GLOSSARY_TITLE_TOKENS
                    )
    return forms


def _count_glossary_form(text: str, form: str) -> int:
    return len(
        re.findall(
            rf"(?<![A-Za-zÀ-ÿ]){re.escape(form)}(?![A-Za-zÀ-ÿ])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _changes_protected_glossary_form(
    original: str,
    replacement: str,
    glossary_terms: list[dict] | None,
) -> bool:
    return any(
        _count_glossary_form(original, form) != _count_glossary_form(replacement, form)
        for form in _protected_glossary_forms(glossary_terms)
    )


def _capitalized_tokens(value: str) -> set[str]:
    return {
        match.group("token").casefold()
        for match in _CAPITALIZED_TOKEN_RE.finditer(value)
    }


def _sentence_initial_capitalized_tokens(value: str) -> set[str]:
    """Return title-cased words that merely begin a sentence in running prose."""
    tokens: set[str] = set()
    for match in _CAPITALIZED_TOKEN_RE.finditer(value):
        preceding_clause = re.split(r"[.!?…\n]", value[: match.start()])[-1]
        if not re.search(r"[A-Za-zÀ-ÿ]", preceding_clause):
            tokens.add(match.group("token").casefold())
    return tokens


def _glossary_capitalized_tokens(terms: list[dict] | None) -> set[str]:
    """Return capitalized glossary tokens that can safely appear in an edit."""
    tokens: set[str] = set()
    for term in terms or []:
        values: list[Any] = [term.get("key"), term.get("pt")]
        for field in (
            "source_aliases",
            "aliases",
            "allowed_target_aliases",
        ):
            raw = term.get(field) or []
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, list):
                values.extend(raw)
        for value in values:
            tokens.update(_capitalized_tokens(str(value or "")))
    return tokens


def _introduces_unknown_capitalized_token(
    original: str,
    replacement: str,
    source_text: str,
    glossary_terms: list[dict] | None,
) -> bool:
    """Reject an edit that invents a name the source/glossary cannot support."""
    introduced = _capitalized_tokens(replacement) - _capitalized_tokens(original)
    if not introduced:
        return False
    introduced.difference_update(_sentence_initial_capitalized_tokens(replacement))
    if not introduced:
        return False
    allowed = _capitalized_tokens(source_text)
    allowed.update(_glossary_capitalized_tokens(glossary_terms))
    return bool(introduced - allowed)


def _gendered_character_variants(
    terms: list[dict] | None,
) -> list[tuple[bool, set[str]]]:
    """Return feminine/masculine character name variants from the glossary."""
    entries: list[tuple[bool, set[str]]] = []
    for term in terms or []:
        gender = str(term.get("gender") or "").strip().casefold()
        category = str(
            term.get("category") or term.get("type") or term.get("term_type") or ""
        ).casefold()
        if not gender.startswith(("femin", "mascul")) or not any(
            marker in category
            for marker in ("person", "personagem", "criatura", "monstro", "monster")
        ):
            continue
        variants: set[str] = set()
        values: list[Any] = [term.get("key"), term.get("pt")]
        for field in (
            "source_aliases",
            "aliases",
            "allowed_target_aliases",
        ):
            raw = term.get(field) or []
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, list):
                values.extend(raw)
        for value in values:
            clean = str(value or "").strip()
            if not clean:
                continue
            variants.add(clean)
            variants.update(
                token
                for token in clean.split()
                if len(token) >= 3 and token[:1].isupper()
            )
        if variants:
            entries.append((gender.startswith("femin"), variants))
    return entries


def _gendered_participles_near_name(text: str, name: str) -> set[tuple[str, str]]:
    """Return gendered participle stems in the clause after a known name."""
    findings: set[tuple[str, str]] = set()
    name_re = re.compile(
        rf"(?<![A-Za-zÀ-ÿ]){re.escape(name)}(?![A-Za-zÀ-ÿ])",
        flags=re.IGNORECASE,
    )
    for name_match in name_re.finditer(text):
        window = text[name_match.end() : name_match.end() + 120]
        sentence_end = re.search(r"[.!?…\n]", window)
        if sentence_end:
            window = window[: sentence_end.start()]
        for match in _GENDERED_PARTICIPLE_RE.finditer(window):
            ending = match.group("ending").casefold()
            gender = "f" if ending.endswith(("ada", "ida")) else "m"
            findings.add((match.group("stem").casefold(), gender))
    return findings


def _changes_known_gender_inflection(
    original: str,
    replacement: str,
    glossary_terms: list[dict] | None,
) -> bool:
    """Keep a reviewer from flipping a known character's grammatical gender."""
    for feminine, variants in _gendered_character_variants(glossary_terms):
        for variant in variants:
            before = _gendered_participles_near_name(original, variant)
            after = _gendered_participles_near_name(replacement, variant)
            for stem, expected in before:
                opposite = "m" if expected == "f" else "f"
                if (stem, opposite) not in after:
                    continue
                if feminine and expected == "f":
                    return True
                if not feminine and expected == "m":
                    return True
    return False


_AGREEMENT_TOKEN_PAIRS = {
    ("o", "a"),
    ("a", "o"),
    ("os", "as"),
    ("as", "os"),
    ("um", "uma"),
    ("uma", "um"),
    ("uns", "umas"),
    ("umas", "uns"),
    ("seu", "sua"),
    ("sua", "seu"),
    ("seus", "suas"),
    ("suas", "seus"),
    ("pelo", "pela"),
    ("pela", "pelo"),
    ("pelos", "pelas"),
    ("pelas", "pelos"),
}


def _is_basic_agreement_correction(original: str, replacement: str) -> bool:
    """Allow small article/pronoun agreement repairs through the synonym gate."""
    token_pattern = r"[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)?"
    before = re.findall(token_pattern, original.casefold())
    after = re.findall(token_pattern, replacement.casefold())
    if len(before) != len(after) or not before:
        return False
    changed = [pair for pair in zip(before, after) if pair[0] != pair[1]]
    return bool(changed) and all(pair in _AGREEMENT_TOKEN_PAIRS for pair in changed)


def _changes_collective_subject_agreement(original: str, replacement: str) -> bool:
    """Keep accepted singular agreement after ``a maioria`` intact."""
    return bool(
        _COLLECTIVE_SINGULAR_PREDICATE_RE.search(original)
        and _COLLECTIVE_PLURAL_PREDICATE_RE.search(replacement)
    )


def _loses_brainwash_meaning(original: str, replacement: str) -> bool:
    """A malformed brainwashing phrase must be repaired, not weakened."""
    return bool(
        _BRAINWASH_CALQUE_RE.search(original)
        and not _BRAINWASH_CANONICAL_RE.search(replacement)
    )


def _changes_esquecer_regency_style(original: str, replacement: str) -> bool:
    """Do not rewrite the natural PT-BR ``esquecer de`` construction by style."""
    before = re.search(r"\bnão\s+esqueci\s+d[ao]s?\s+", original, re.IGNORECASE)
    after = re.search(r"\bnão\s+esqueci\s+[ao]s?\s+", replacement, re.IGNORECASE)
    return bool(before and after)


def _changes_feminine_member_article(original: str, replacement: str) -> bool:
    """Avoid silently turning a feminine addressee into masculine ``membro``."""
    return bool(
        re.search(r"\buma\s+membro\b", original, re.IGNORECASE)
        and re.search(r"\bum\s+membro\b", replacement, re.IGNORECASE)
    )


def _introduces_lembrei_pronominal_style(original: str, replacement: str) -> bool:
    """Keep a natural ``Lembrei`` form from becoming more formal by preference."""
    return bool(
        _LEMBREI_PRONOMINAL_RE.search(replacement)
        and not _LEMBREI_PRONOMINAL_RE.search(original)
    )


def _changes_lembrei_natural_regency_style(
    original: str, replacement: str
) -> bool:
    """Preserve the natural PT-BR ``Lembrei de/das`` construction."""
    return bool(
        _LEMBREI_WITH_DE_RE.search(original)
        and _LEMBREI_DIRECT_OBJECT_RE.search(replacement)
    )


def _formalizes_colloquial_direct_object(
    original: str, replacement: str
) -> bool:
    """Keep natural post-verbal tonic objects out of prescriptive editing."""
    clitics = {"ele": "o", "ela": "a", "eles": "os", "elas": "as"}
    for before in _COLLOQUIAL_DIRECT_OBJECT_RE.finditer(original):
        expected_clitic = clitics[before.group("object").casefold()]
        expected_verb = before.group("verb").casefold()
        for after in _FORMAL_PREVERBAL_CLITIC_RE.finditer(replacement):
            if (
                after.group("clitic").casefold() == expected_clitic
                and after.group("verb").casefold() == expected_verb
            ):
                return True
    return False


def _formalizes_colloquial_pt_br_register(original: str, replacement: str) -> bool:
    """Reject narrowly-defined grammar edits that only make dialogue formal."""
    if _formalizes_colloquial_direct_object(original, replacement):
        return True
    if (
        _COLLOQUIAL_AGRADECER_VOCE_RE.search(original)
        and _FORMAL_AGRADECER_CLITIC_RE.search(replacement)
    ):
        return True
    return bool(
        _COLLOQUIAL_DEIXAR_EU_RE.search(original)
        and _FORMAL_DEIXAR_QUE_EU_RE.search(replacement)
    )


def _is_low_signal_single_token_substitution(original: str, replacement: str) -> bool:
    """Reject a one-word synonym swap that cannot be verified structurally."""
    if _is_basic_agreement_correction(original, replacement):
        return False
    token_pattern = r"[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)?"
    original_tokens = re.findall(token_pattern, original.casefold())
    replacement_tokens = re.findall(token_pattern, replacement.casefold())
    if len(original_tokens) != len(replacement_tokens) or not original_tokens:
        return False
    changed = [
        (before, after)
        for before, after in zip(original_tokens, replacement_tokens)
        if before != after
    ]
    if len(changed) != 1:
        return False
    before, after = changed[0]
    return SequenceMatcher(a=before, b=after).ratio() < 0.60


def _has_no_meaningful_token_overlap(original: str, replacement: str) -> bool:
    """Reject an unanchored rewrite that is unlikely to be a minimal edit."""
    token_pattern = r"[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)?"

    def meaningful_tokens(value: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(token_pattern, value)
            if token.casefold() not in _REVIEW_OVERLAP_STOPWORDS
        }

    before = meaningful_tokens(original)
    after = meaningful_tokens(replacement)
    return bool(len(before) >= 2 and len(after) >= 2 and not (before & after))


def _introduces_reviewer_grammar_defect(original: str, replacement: str) -> str | None:
    """Reject recurring malformed reviewer outputs before they reach final QA."""
    checks = (
        (_DANGLING_PREPOSITION_RE, "review_dangling_preposition"),
        (_INVALID_BRAINWASH_AUXILIARY_RE, "review_brainwash_auxiliary"),
        (_DUPLICATE_JOINED_TOGETHER_RE, "review_duplicate_joined_together"),
        (_RECOVER_CONSCIOUSNESS_REFLEXIVE_RE, "review_recover_consciousness"),
        (_DUPLICATE_RECOVER_REFLEXIVE_RE, "review_duplicate_recover_reflexive"),
        (
            _RECOVER_SUBJUNCTIVE_AFTER_PARA_QUE_RE,
            "review_recover_subjunctive",
        ),
    )
    for pattern, reason in checks:
        if pattern.search(replacement) and not pattern.search(original):
            return reason
    return None


def _introduces_periphrastic_perfect(
    original: str,
    replacement: str,
) -> bool:
    """Reject a risky literal English-present-perfect calque in PT-BR.

    The reviewer cannot safely infer iterative aspect from an isolated source
    chunk. It must not turn a natural completed-action preterite into
    ``ter + participio`` merely because English used a perfect form.
    """
    return bool(
        _PT_PERIPHRASTIC_PERFECT_RE.search(replacement)
        and not _PT_PERIPHRASTIC_PERFECT_RE.search(original)
    )


def _changes_dialogue_treatment(original: str, replacement: str) -> bool:
    """Keep deliberate ``tu``/``você`` variation out of copy-editing."""
    original_clitics = {
        match.group("clitic").casefold()
        for match in _DIALOGUE_TREATMENT_CLITIC_RE.finditer(original)
    }
    replacement_clitics = {
        match.group("clitic").casefold()
        for match in _DIALOGUE_TREATMENT_CLITIC_RE.finditer(replacement)
    }
    if not original_clitics or not replacement_clitics:
        return False
    tu_forms = {"te", "vos"}
    voce_forms = {"se", "nos"}
    return bool(
        (original_clitics & tu_forms and replacement_clitics & voce_forms)
        or (original_clitics & voce_forms and replacement_clitics & tu_forms)
    )


def _is_short_phrase_reorder(original: str, replacement: str) -> bool:
    """Reject a partial noun-phrase reorder that may break outside agreement."""
    token_pattern = r"[A-Za-zÀ-ÿ]+"
    original_tokens = [
        token.casefold()
        for token in re.findall(token_pattern, original)
        if token.casefold() not in _SHORT_PHRASE_STOPWORDS
    ]
    replacement_tokens = [
        token.casefold()
        for token in re.findall(token_pattern, replacement)
        if token.casefold() not in _SHORT_PHRASE_STOPWORDS
    ]
    return bool(
        2 <= len(original_tokens) <= 3
        and Counter(original_tokens) == Counter(replacement_tokens)
        and original_tokens != replacement_tokens
    )


def _validate_suggestion(
    *,
    source_text: str,
    current_text: str,
    original: str,
    replacement: str,
    reason: str,
    glossary_terms: list[dict] | None = None,
) -> str | None:
    if not reason:
        return "review_missing_reason"
    if len(original) < 4:
        return "review_original_too_short"
    if "\n" in original or "\n" in replacement:
        return "review_multiline_change"
    if original == replacement:
        return "review_noop"
    if _is_optional_proper_name_article_change(original, replacement):
        return "review_optional_proper_name_article"
    if _changes_protected_glossary_form(original, replacement, glossary_terms):
        return "review_changed_protected_glossary_form"
    if _changes_dialogue_treatment(original, replacement):
        return "review_changed_dialogue_treatment"
    if _changes_collective_subject_agreement(original, replacement):
        return "review_collective_singular_agreement"
    if _changes_esquecer_regency_style(original, replacement):
        return "review_esquecer_regency_style"
    if _introduces_lembrei_pronominal_style(original, replacement):
        return "review_lembrei_registro"
    if _changes_lembrei_natural_regency_style(original, replacement):
        return "review_lembrei_registro"
    if _formalizes_colloquial_pt_br_register(original, replacement):
        return "review_colloquial_register"
    if _changes_feminine_member_article(original, replacement):
        return "review_changed_feminine_member"
    if _introduces_unknown_capitalized_token(
        original, replacement, source_text, glossary_terms
    ):
        return "review_introduced_unknown_proper_name"
    if _changes_known_gender_inflection(original, replacement, glossary_terms):
        return "review_changed_known_gender"
    if _BRAINWASH_CALQUE_RE.search(replacement):
        return "review_brainwash_calque"
    if _loses_brainwash_meaning(original, replacement):
        return "review_brainwash_semantic_loss"
    if (
        _LITERAL_MADE_MOVE_RE.search(replacement)
        and not _LITERAL_MADE_MOVE_RE.search(original)
    ):
        return "review_literal_move_calque"
    if (
        _UNGRAMMATICAL_TINGIR_SUBJUNCTIVE_RE.search(replacement)
        and not _UNGRAMMATICAL_TINGIR_SUBJUNCTIVE_RE.search(original)
    ):
        return "review_ungrammatical_subjunctive"
    if (
        _RECOVER_FROM_CALQUE_RE.search(replacement)
        and not _RECOVER_FROM_CALQUE_RE.search(original)
    ):
        return "review_recover_from_calque"
    grammar_defect = _introduces_reviewer_grammar_defect(original, replacement)
    if grammar_defect:
        return grammar_defect
    if _has_no_meaningful_token_overlap(original, replacement):
        return "review_unanchored_rewrite"
    if _is_low_signal_single_token_substitution(original, replacement):
        return "review_single_token_synonym"
    if _introduces_periphrastic_perfect(original, replacement):
        return "review_periphrastic_perfect_calque"
    if _is_short_phrase_reorder(original, replacement):
        return "review_short_phrase_reorder"
    if current_text.count(original) != 1:
        return "review_original_not_unique"
    if len(original) >= 20:
        ratio = len(replacement) / max(len(original), 1)
        if ratio < 0.45 or ratio > 2.2:
            return f"review_replacement_ratio:{ratio:.2f}"
    if count_quotes(original) != count_quotes(replacement):
        return "review_changed_quote_count"
    if "###" in replacement or '"""' in replacement or "'''" in replacement:
        return "review_format_marker"
    return None


def validate_bilingual_review_candidate(
    *,
    source_text: str,
    translated_text: str,
    candidate_text: str,
    glossary_terms: list[dict] | None = None,
) -> str | None:
    """Reject a candidate that regresses structure or deterministic QA."""
    current = translated_text.strip()
    candidate = candidate_text.strip()
    if not candidate:
        return "review_empty"
    if not current:
        return None
    ratio = len(candidate) / max(len(current), 1)
    if len(current) >= 200 and (ratio < 0.85 or ratio > 1.18):
        return f"review_output_ratio:{ratio:.2f}"
    if _paragraph_count(candidate) != _paragraph_count(current):
        return "review_changed_paragraph_count"
    if count_quotes(candidate) != count_quotes(current):
        return "review_changed_dialogue_count"
    current_english, _ = detect_residual_english(current)
    candidate_english, _ = detect_residual_english(candidate)
    if candidate_english and not current_english:
        return "review_introduced_residual_english"
    if detect_entity_mutation(current, candidate):
        return "review_entity_mutation"
    if _changes_protected_glossary_form(current, candidate, glossary_terms):
        return "review_changed_protected_glossary_form"
    if _changes_collective_subject_agreement(current, candidate):
        return "review_collective_singular_agreement"
    if _changes_esquecer_regency_style(current, candidate):
        return "review_esquecer_regency_style"
    if _introduces_lembrei_pronominal_style(current, candidate):
        return "review_lembrei_registro"
    if _changes_lembrei_natural_regency_style(current, candidate):
        return "review_lembrei_registro"
    if _changes_feminine_member_article(current, candidate):
        return "review_changed_feminine_member"
    if _introduces_unknown_capitalized_token(
        current, candidate, source_text, glossary_terms
    ):
        return "review_introduced_unknown_proper_name"
    if _changes_known_gender_inflection(current, candidate, glossary_terms):
        return "review_changed_known_gender"
    if (
        _BRAINWASH_CALQUE_RE.search(candidate)
        and not _BRAINWASH_CALQUE_RE.search(current)
    ):
        return "review_brainwash_calque"
    if (
        _LITERAL_MADE_MOVE_RE.search(candidate)
        and not _LITERAL_MADE_MOVE_RE.search(current)
    ):
        return "review_literal_move_calque"
    if (
        _UNGRAMMATICAL_TINGIR_SUBJUNCTIVE_RE.search(candidate)
        and not _UNGRAMMATICAL_TINGIR_SUBJUNCTIVE_RE.search(current)
    ):
        return "review_ungrammatical_subjunctive"
    if (
        _RECOVER_FROM_CALQUE_RE.search(candidate)
        and not _RECOVER_FROM_CALQUE_RE.search(current)
    ):
        return "review_recover_from_calque"
    grammar_defect = _introduces_reviewer_grammar_defect(current, candidate)
    if grammar_defect:
        return grammar_defect
    if detect_model_collapse(candidate, original_len=len(current), mode="refine"):
        return "review_model_collapse"

    terms = glossary_terms or []
    current_quality = run_translation_quality_checks(source_text, current, terms)
    candidate_quality = run_translation_quality_checks(source_text, candidate, terms)
    if candidate_quality["score"] < current_quality["score"]:
        return "review_qa_score_regression"
    if candidate_quality["issue_count"] > current_quality["issue_count"]:
        return "review_qa_issue_regression"
    return None


def review_translation_chunk(
    *,
    source_text: str,
    translated_text: str,
    backend: LLMBackend,
    logger: logging.Logger,
    glossary_text: str | None = None,
    glossary_terms: list[dict] | None = None,
    max_changes: int = DEFAULT_MAX_CHANGES,
    cache_metadata: dict[str, Any] | None = None,
) -> BilingualReviewResult:
    """Request, validate, and apply minimal source-aware copy edits."""
    started = time.perf_counter()
    current = translated_text.strip()
    focus_issues = detect_bilingual_review_focuses(current)
    if not source_text.strip() or not current:
        return BilingualReviewResult(
            text=translated_text,
            elapsed_seconds=time.perf_counter() - started,
        )

    metadata = {
        "mode": "bilingual_review",
        "pipeline_version": BILINGUAL_REVIEW_PIPELINE_VERSION,
        "prompt_hash": bilingual_review_prompt_fingerprint(),
        "backend": getattr(backend, "backend", None),
        "model": getattr(backend, "model", None),
        "num_predict": getattr(backend, "num_predict", None),
        "temperature": getattr(backend, "temperature", None),
        "repeat_penalty": getattr(backend, "repeat_penalty", None),
        "seed": getattr(backend, "seed", None),
        "max_changes": max_changes,
        "focus_issues": focus_issues,
        **(cache_metadata or {}),
    }
    cache_key = chunk_hash(
        json.dumps(
            {
                "source": source_text,
                "translated": current,
                "glossary": glossary_text or "",
                "metadata": metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if cache_exists("review", cache_key):
        cached = load_cache("review", cache_key)
        cached_text = str(cached.get("final_output") or "").strip()
        cached_meta = cached.get("metadata") if isinstance(cached.get("metadata"), dict) else {}
        if cached_text and not validate_bilingual_review_candidate(
            source_text=source_text,
            translated_text=current,
            candidate_text=cached_text,
            glossary_terms=glossary_terms,
        ):
            applied = cached_meta.get("applied_changes") or []
            return BilingualReviewResult(
                text=cached_text,
                attempted=True,
                changed=cached_text != current,
                used_cache=True,
                suggested_changes=int(cached_meta.get("suggested_changes", len(applied)) or 0),
                applied_changes=applied if isinstance(applied, list) else [],
                elapsed_seconds=time.perf_counter() - started,
                focus_issues=focus_issues,
            )

    prompt = build_bilingual_review_prompt(
        source_text=source_text,
        translated_text=current,
        glossary_text=glossary_text,
        focus_issues=focus_issues,
        max_changes=max_changes,
    )
    try:
        raw = backend.generate(prompt).text
    except Exception as exc:
        logger.warning("Revisao bilingue falhou; mantendo traducao atual: %s", exc)
        return BilingualReviewResult(
            text=translated_text,
            attempted=True,
            llm_attempts=1,
            failure_reason="review_backend_error",
            elapsed_seconds=time.perf_counter() - started,
        )

    try:
        suggestions = parse_bilingual_review_output(raw, max_changes=max_changes)
    except ValueError as exc:
        return BilingualReviewResult(
            text=translated_text,
            attempted=True,
            llm_attempts=1,
            failure_reason=str(exc),
            raw_output=raw,
            elapsed_seconds=time.perf_counter() - started,
        )

    candidate = current
    applied: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for suggestion in suggestions:
        original = suggestion["original"]
        replacement = suggestion["substituicao"]
        reason = _validate_suggestion(
            source_text=source_text,
            current_text=candidate,
            original=original,
            replacement=replacement,
            reason=suggestion["motivo"],
            glossary_terms=glossary_terms,
        )
        if reason:
            rejected.append({**suggestion, "rejeitada_por": reason})
            continue
        proposed = candidate.replace(original, replacement, 1)
        reason = validate_bilingual_review_candidate(
            source_text=source_text,
            translated_text=current,
            candidate_text=proposed,
            glossary_terms=glossary_terms,
        )
        if reason:
            rejected.append({**suggestion, "rejeitada_por": reason})
            continue
        candidate = proposed
        applied.append(suggestion)

    final_reason = validate_bilingual_review_candidate(
        source_text=source_text,
        translated_text=current,
        candidate_text=candidate,
        glossary_terms=glossary_terms,
    )
    if final_reason:
        candidate = current
        rejected.append({"rejeitada_por": final_reason})
        applied = []

    cache_metadata_payload = {
        **metadata,
        "suggested_changes": len(suggestions),
        "applied_changes": applied,
        "rejected_changes": rejected,
    }
    save_cache("review", cache_key, raw, candidate, cache_metadata_payload)
    return BilingualReviewResult(
        text=candidate,
        attempted=True,
        changed=candidate != current,
        llm_attempts=1,
        suggested_changes=len(suggestions),
        applied_changes=applied,
        rejected_changes=rejected,
        raw_output=raw,
        elapsed_seconds=time.perf_counter() - started,
        focus_issues=focus_issues,
    )
