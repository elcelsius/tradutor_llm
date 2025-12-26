from __future__ import annotations

import re

_BIRTH_CONTEXT = re.compile(r"\b(beb[eê]|filh[oa]s?|gravidez|grávida|gravida|gesta[cç][aã]o|matern|parto|bebes)\b", re.IGNORECASE)


def postprocess_translation(pt_text: str, en_text: str | None = None) -> str:
    """
    Ajustes determinísticos pós-tradução para falsos cognatos comuns.

    - Se o original contém parry/parried/parrying e a tradução trouxe parriu/parrir/parrindo/etc.,
      substitui por formas de "aparar".
    - Evita interferir em contextos de parto (bebê/gravidez/etc).
    """
    if not pt_text:
        return pt_text
    if en_text is None or not re.search(r"\bparr(?:y|ied|ying)\b", en_text, flags=re.IGNORECASE):
        return pt_text
    if _BIRTH_CONTEXT.search(pt_text):
        return pt_text

    def _replace(match: re.Match[str]) -> str:
        suffix = match.group(1).lower()
        if suffix in {"r", "ir"}:
            return "aparar"
        if suffix in {"u", "iu", "ou"}:
            return "aparou"
        if suffix in {"ndo"}:
            return "aparando"
        if suffix in {"ido", "ida", "idos", "idas"}:
            base = "aparad"
            end = suffix[2:]  # o / a / os / as
            return f"{base}{end}"
        if suffix in {"ia", "iam"}:
            return "aparava" if suffix == "ia" else "aparavam"
        return "aparou"

    pattern = re.compile(r"\bparr(iu|ir|indo|ido|ida|idos|idas|ia|iam|ou|r)\b", flags=re.IGNORECASE)
    return pattern.sub(_replace, pt_text)
