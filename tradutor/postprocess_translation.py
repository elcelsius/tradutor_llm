from __future__ import annotations

import re

from .quote_fix import collapse_repeated_curly_quotes
from .text_postprocess import translate_embedded_english_connectors

_BIRTH_CONTEXT = re.compile(
    r"\b(bebe|bebes|bebê|bebês|filh[oa]s?|gravidez|gr[aá]vida|gesta[cç][aã]o|matern|parto)\b",
    re.IGNORECASE,
)
_DASH_TRAIL_QUOTE_RE = re.compile(r"^(—\s.*?)[\"”](?=[,;.\s]|$)")
_DASH_LEAD_QUOTE_RE = re.compile(r"^(—\s*)[\"“]\s*(.+)$")
_DASH_SPEECH_TAG_RE = re.compile(
    r"^(—\s[^\"”]{0,120}?)[\"”]([\s,;.!?]+[A-Z\u00c0-\u017f].*)$"
)

# Um ``I`` isolado antes de uma palavra PT-BR pode escapar de uma resposta
# parcialmente traduzida. A lista deliberadamente curta evita transformar uma
# frase inteiramente em ingles, como ``I have no desire to die``, em um hibrido
# ainda mais dificil de detectar (``Eu have ...``).
_STRAY_I_BEFORE_PORTUGUESE_RE = re.compile(
    r"(?<![A-Za-zÀ-ÿ])I\s+(?=(?:"
    r"não|nao|já|ja|me|te|lhe|o|a|os|as|um|uma|isso|aquilo|"
    r"aqui|ali|acho|aposto|sei|quero|vou|posso|preciso|entendo|penso|"
    r"sinto|tenho|estou|era|fui|gosto|odeio"
    r")\b)",
    re.IGNORECASE,
)
_ELLIPSIS_LOWERCASE_CONTINUATION_RE = re.compile(
    r"(?P<ellipsis>\.\.\.|…)[ \t]*\n[ \t]*\n+(?P<continuation>[a-zà-ÿ])"
)
_CARRIAGE_SOURCE_RE = re.compile(r"\bcarriages?\b", re.IGNORECASE)


def _canonicalize_carriage_artifacts(text: str, source_text: str | None) -> str:
    """Corrige carro/carroça somente em chunks cuja fonte fala de carruagens."""
    if not source_text or not _CARRIAGE_SOURCE_RE.search(source_text):
        return text

    replacements = (
        (r"\bdentro daquele carro\b", "dentro daquela carruagem"),
        (r"\bpara o carroça\b", "para a carruagem"),
        (r"\bpara o carro\b", "para a carruagem"),
        (r"\bnos carros\b", "nas carruagens"),
        (r"\bno carro\b", "na carruagem"),
        (r"\bdo carro\b", "da carruagem"),
        (r"\bcarros de\b", "carruagens de"),
        (r"\bcarro de\b", "carruagem de"),
        (r"\bdo carroça\b", "da carruagem"),
        (r"\bno carroça\b", "na carruagem"),
        (r"\bcarroças\b", "carruagens"),
        (r"\bcarroça\b", "carruagem"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def postprocess_translation(pt_text: str, en_text: str | None = None) -> str:
    """
    Ajustes determinísticos pós-tradução para falsos cognatos comuns e artefatos de fala.

    - Se o original contém parry/parried/parrying e a tradução trouxe parriu/parrir/parrindo/etc.,
      substitui por formas de "aparar".
    - Evita interferir em contextos de parto (bebê/gravidez/etc).
    - Remove aspas sobrando em falas que começam com travessão.
    """
    if not pt_text:
        return pt_text

    pt_text, _ = collapse_repeated_curly_quotes(pt_text)
    # Alguns retornos do modelo duplicam o fechamento tipográfico com uma aspa
    # reta solta (por exemplo, ``?” "``). Não é uma citação aninhada porque as
    # duas aspas estão adjacentes; remover o artefato preserva o diálogo válido.
    pt_text = re.sub(r'([“”])[ \t]*"', r"\1", pt_text)
    pt_text = re.sub(r'"[ \t]*([“”])', r"\1", pt_text)
    pt_text = translate_embedded_english_connectors(pt_text)
    pt_text = _canonicalize_carriage_artifacts(pt_text, en_text)

    # A extração pode deixar uma continuação de frase em linha própria após
    # reticências. Quando a LLM a transforma em parágrafo, o texto passa a
    # iniciar uma narração com minúscula. É uma fronteira inequivocamente
    # artificial, então restaura-se a continuidade antes da QA.
    pt_text = _ELLIPSIS_LOWERCASE_CONTINUATION_RE.sub(
        lambda match: f"{match.group('ellipsis')} {match.group('continuation')}",
        pt_text,
    )

    # Corrige resíduos híbridos do inglês antes da QA por chunk. Se forem
    # deixados para a revisão final, o guardrail os rejeita repetidamente.
    pt_text = re.sub(r"(?<![A-Za-zÀ-ÿ])I-isso\b", "S-sim", pt_text, flags=re.IGNORECASE)
    pt_text = _STRAY_I_BEFORE_PORTUGUESE_RE.sub("Eu ", pt_text)
    pt_text = re.sub(r"\buh+\b", "Ah", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(r"\barright\b", "Beleza", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(r"\bboost\b", "impulso", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(r"\bthey\s+todos\b", "todos", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(
        r"\bY-you\s+divindades\s+podem\b",
        "V-vocês, divindades, podem",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bV-você\s+divinos\s+podem\b",
        "V-vocês, divindades, podem",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(r"\bY-you\b", "V-você", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(
        r"\b(super\s+)?desconfiad([ao])\s+AF\b",
        r"\1desconfiad\2 pra caramba",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(r"\bthough\b", "porém", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(r"\btã\b", "tão", pt_text, flags=re.IGNORECASE)
    pt_text = re.sub(
        r"\bA presidente da classe tá dormindo como uma princesa e tão bonita fazendo isso\b",
        "A presidente da classe tá dormindo como uma princesa e tá tão bonita assim",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bA algum momento depois de\b",
        "Em algum momento depois de",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bConvincentemente Nyantan a confiar nela\b",
        "Convencer Nyantan a confiar nela",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bYasu fora um dos vários que pegaram no caminho\b",
        "Yasu fora um dos vários que eles haviam levado consigo pelo caminho",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bTerei a maior dor sobre todas as dimensões e todos os mundos\b",
        "Trarei o maior sofrimento sobre todas as dimensões e todos os mundos",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bTragarei o maior sofrimento sobre todas as dimensões e todos os mundos\b",
        "Trarei o maior sofrimento sobre todas as dimensões e todos os mundos",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bForra uma criatura fraca\b",
        "Era uma criatura fraca",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\btrajes de espadachins voadoras\b",
        "trajes de espadachins voadores",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bAh, Pode calar a boca\b",
        "Ah, pode calar a boca",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bdivindades mais significativas fossem enviadas\b",
        "divindades mais poderosas fossem enviadas",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bvirando (?:o|seu) ódio uns contra os outros\b",
        "voltando seu ódio uns contra os outros",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bchutou-a levemente quando ela caiu\b",
        "chutou-a de leve de volta para o alto quando ela caiu",
        pt_text,
        flags=re.IGNORECASE,
    )
    pt_text = re.sub(
        r"\bchutou levemente de volta quando ela caiu\b",
        "chutou-a de leve de volta para o alto quando ela caiu",
        pt_text,
        flags=re.IGNORECASE,
    )
    if en_text and re.search(
        r"\bThere is one (?:matter|thing) on my mind\b", en_text, flags=re.IGNORECASE
    ):
        pt_text = re.sub(
            r"\bHá uma coisa em mente\b",
            "Há uma coisa que está me preocupando",
            pt_text,
            flags=re.IGNORECASE,
        )

    # Parry/parried/parrying -> aparar (somente quando presente no EN e fora de contexto de parto)
    if (
        en_text is not None
        and re.search(r"\bparr(?:y|ied|ying)\b", en_text, flags=re.IGNORECASE)
        and not _BIRTH_CONTEXT.search(pt_text)
    ):

        def _replace(match: re.Match[str]) -> str:
            """Processamento interno auxiliar."""
            suffix = match.group(1).lower()
            if suffix in {"r", "ir"}:
                return "aparar"
            if suffix in {"u", "iu", "ou"}:
                return "aparou"
            if suffix in {"ndo"}:
                return "aparando"
            if suffix in {"ido", "ida", "idos", "idas"}:
                base = "aparad"
                end = suffix[2:]
                return f"{base}{end}"
            if suffix in {"ia", "iam"}:
                return "aparava" if suffix == "ia" else "aparavam"
            return "aparou"

        pattern = re.compile(
            r"\bparr(iu|ir|indo|ido|ida|idos|idas|ia|iam|ou|r)\b", flags=re.IGNORECASE
        )
        pt_text = pattern.sub(_replace, pt_text)

    # Limpeza de travessão + aspas mistas
    fixed_lines: list[str] = []
    for ln in pt_text.splitlines():
        cleaned = ln.strip()
        cleaned = _DASH_TRAIL_QUOTE_RE.sub(r"\1", cleaned)
        cleaned = _DASH_LEAD_QUOTE_RE.sub(r"\1\2", cleaned)
        cleaned = _DASH_SPEECH_TAG_RE.sub(r"\1\2", cleaned)
        fixed_lines.append(cleaned if cleaned != "" else ln)

    return "\n".join(fixed_lines)
