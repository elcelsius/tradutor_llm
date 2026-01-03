"""
Pré-processamento de PDFs: extração de texto, limpeza e chunking seguro.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from collections import Counter
from typing import Final, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - fallback para ambientes sem PyMuPDF
    class _DummyDoc:
        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            return False

        def __iter__(self):
            return iter([])

        @property
        def pages(self):
            return []

        def __len__(self):
            return 0

    class _DummyFitz:
        def open(self, *args, **kwargs):
            return _DummyDoc()

    fitz = _DummyFitz()  # type: ignore

from .utils import chunk_by_paragraphs

# Watermarks de sites/grupos de scan.
FOOTER_PATTERNS: Final[list[str]] = [
    r"\bPage\s+\d+\b",
    r"Goldenagato \| mp4directs\.com",
    r"mp4directs\.com",
    r"zerobooks?",
    r"jnovels?",
    r"newsletter",
    r"discord\.gg",
    r"stay up to date",
    r"download(?:ing)? our mobile app",
]

NOISE_PARAGRAPH_PATTERNS: Final[list[str]] = [
    r"stay up to date",
    r"download(?:ing)? our mobile app",
    r"zerobooks",
    r"jnovels",
    r"join our discord",
    r"newsletter",
    r"follow us",
    r"support us",
    r"read (more|the latest) on",
    r"get the latest news",
    r"visit us online",
    r"visit us",
]

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
URL_RE = re.compile(r"(https?://|www\.)|(\w+\.(?:com|net|org|io|gg|co|me|xyz)(?:[/?#]|$))", re.IGNORECASE)

TOC_MARKER_RE = re.compile(
    r"^(prologue|epilogue|afterword|chapter\s+\d+(?::[^\n]+)?)\s*$",
    re.IGNORECASE,
)

PROMO_DOMAINS: Final[list[str]] = [
    "oceanofpdf",
    "gomanga.com",
    "jnovels",
    "zerobooks",
    "discord.gg",
    "patreon",
]

PROMO_PHRASES: Final[list[str]] = [
    "sign up for",
    "read online",
    "download",
    "join our",
    "support us",
    "read more on",
    "follow us",
    "thank you for reading",
    "thank you for downloading",
    "visit us online",
    "get the latest news",
]

TOC_MARKER_LINES: Final[list[str]] = [
    "table of contents",
    # "contents" tratado como marker apenas quando a linha é exatamente o termo, ver _is_marker
    "sumário",
    "índice",
    "indice",
    "índice remissivo",
    "color inserts",
    "inserções coloridas",
    "title page",
    "página de título",
    "copyrights and credits",
    "newsletter",
]


def normalize_line_for_filters(line: str) -> str:
    """Normaliza uma linha para fins de filtro (não altera texto final)."""
    if not line:
        return ""
    normalized = line.replace("\xa0", " ")
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = normalized.strip()
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_noise_glossary() -> dict:
    return {
        "line_contains": PROMO_DOMAINS + PROMO_PHRASES,
        "line_compact_contains": ["oceanofpdf", "zerobooks", "jnovels", "gomanga", "discordgg", "patreon"],
        "line_regex": [],
        "max_line_len": 160,
    }


def _load_noise_glossary(path: str | Path | None) -> dict:
    if not path:
        return _default_noise_glossary()
    p = Path(path)
    if not p.exists():
        return _default_noise_glossary()
    try:
        import json

        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _default_noise_glossary()
    merged = _default_noise_glossary()
    for key in ("line_contains", "line_compact_contains", "line_regex"):
        if isinstance(data.get(key), list):
            merged[key] = data[key]
    if isinstance(data.get("max_line_len"), int) and data["max_line_len"] > 0:
        merged["max_line_len"] = data["max_line_len"]
    return merged


def extract_text_from_pdf(path: Path, logger: logging.Logger) -> str:
    """Extrai texto de um PDF usando PyMuPDF."""
    with fitz.open(path) as doc:
        pages = [page.get_text() or "" for page in doc]
    text = "\n".join(pages)
    logger.debug("PDF %s extraído: %d caracteres", path.name, len(text))
    return text


def _remove_headers_footers(text: str) -> str:
    lines = text.splitlines()
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        # Números de página isolados ou cabeçalhos típicos
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if len(stripped) <= 5 and stripped.isupper():
            continue
        if re.search(r"\bpage\b", stripped, re.IGNORECASE):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def sanitize_extracted_text(text: str, logger: Optional[logging.Logger] = None) -> Tuple[str, dict]:
    """
    Remove ruídos determinísticos da extração (caractere U+FFFF e linhas só com números).
    Preserva estrutura de parágrafos.
    """
    stats = {"removed_uffff": 0, "removed_numeric_lines": 0}

    before = text
    text = text.replace("\uFFFF", "").replace("￿", "")
    stats["removed_uffff"] = len(before) - len(text)  # proxy de remoções de char

    lines = text.splitlines()
    cleaned: list[str] = []
    for ln in lines:
        if re.fullmatch(r"\s*\d+\s*", ln):
            stats["removed_numeric_lines"] += 1
            continue
        cleaned.append(ln)
    cleaned_text = "\n".join(cleaned)
    if logger:
        logger.debug(
            "Sanitize extracted: removed_numeric_lines=%d removed_uffff=%d",
            stats["removed_numeric_lines"],
            stats["removed_uffff"],
        )
    return cleaned_text, stats


def _remove_hyphenation(text: str) -> str:
    return re.sub(r"(\w+)-\s*\n(\w+)", r"\1\2\n", text)


def _join_broken_lines(text: str) -> str:
    lines = text.splitlines()
    joined: List[str] = []
    buffer: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                joined.append(" ".join(buffer))
                buffer = []
            continue
        if re.search(r"[.!?…]$", stripped):
            buffer.append(stripped)
            joined.append(" ".join(buffer))
            buffer = []
        else:
            buffer.append(stripped)

    if buffer:
        joined.append(" ".join(buffer))

    return "\n\n".join(joined)


def remove_noise_blocks(text: str) -> str:
    """Remove paragrafos que pareçam ser ads/newsletter/discord etc."""
    paragraphs = text.split("\n\n")
    cleaned: list[str] = []
    for para in paragraphs:
        norm = para.lower().strip()
        if not norm:
            cleaned.append("")
            continue
        if any(re.search(pat, norm, flags=re.IGNORECASE) for pat in NOISE_PARAGRAPH_PATTERNS):
            continue
        cleaned.append(para.strip())
    # reintroduz quebras duplas
    return "\n\n".join(p for p in cleaned if p != "")


def _is_promo_line(line: str) -> bool:
    norm = normalize_line_for_filters(line).lower()
    if not norm:
        return False
    has_domain = any(dom in norm for dom in PROMO_DOMAINS)
    has_phrase = any(phrase in norm for phrase in PROMO_PHRASES)
    return has_domain or has_phrase


def _remove_promo_lines(text: str, glossary: dict) -> tuple[str, dict, list[str]]:
    lines = text.splitlines()
    cleaned: list[str] = []
    stats = {
        "oceanofpdf_removed_count": 0,
        "promo_lines_removed_count": 0,
        "promo_blocks_removed_count": 0,
        "urls_removed_count": 0,
        "promo_samples": [],
        "promo_removed_hash": "",
        "promo_removed_reason_counts": {},
        "promo_removed_samples": {},
    }
    removed_norms: list[str] = []
    max_len = glossary.get("max_line_len", 160) or 160
    line_contains = [s.lower() for s in glossary.get("line_contains", []) if isinstance(s, str)]
    line_compact = [re.sub(r"[^a-z0-9]", "", s.lower()) for s in glossary.get("line_compact_contains", []) if isinstance(s, str)]
    domain_tokens = [s for s in line_contains if "." in s or "/" in s] + line_compact
    phrase_tokens = [s for s in line_contains if s not in domain_tokens]
    regexes = []
    for pat in glossary.get("line_regex", []):
        try:
            regexes.append(re.compile(pat, flags=re.IGNORECASE))
        except re.error:
            continue

    def _is_dialogue_like(norm_line: str, raw_line: str) -> bool:
        stripped = raw_line.lstrip()
        if stripped.startswith(('"', "“", "'", "’", "—", "-")):
            return True
        compact = re.sub(r"[\\s]", "", norm_line)
        if re.fullmatch(r"[.·…]{2,}", norm_line):
            return True
        if len(compact) <= 12 and re.fullmatch(r"[A-Za-z]{1,6}[!?…—.]+", compact):
            return True
        return False

    def _record_reason(reason: str, sample: str) -> None:
        stats["promo_removed_reason_counts"][reason] = stats["promo_removed_reason_counts"].get(reason, 0) + 1
        samples = stats["promo_removed_samples"].setdefault(reason, [])
        if len(samples) < 10:
            samples.append(sample)

    def _update_stats(norm_line: str, *, has_url: bool, has_domain: bool) -> None:
        stats["promo_lines_removed_count"] += 1
        if has_url or has_domain:
            stats["urls_removed_count"] += 1
        if "oceanofpdf" in norm_line:
            stats["oceanofpdf_removed_count"] += 1

    removed_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        normalized = normalize_line_for_filters(line)
        norm_lower = normalized.lower()
        norm_compact = re.sub(r"[^a-z0-9]", "", norm_lower)
        if not normalized:
            cleaned.append("")
            i += 1
            continue
        has_domain = any(dom in norm_lower for dom in domain_tokens) or any(tok in norm_compact for tok in line_compact)
        has_url = bool(URL_RE.search(norm_lower))
        has_phrase = any(phrase in norm_lower for phrase in phrase_tokens)
        has_regex = any(r.search(line) for r in regexes)
        looks_dialogue = line.lstrip().startswith(('"', "“", "'", "’", "-", "–"))
        long_narrative = len(normalized) > 180 and not (has_url or has_domain)
        promo_seed = has_domain or has_url or has_phrase or has_regex
        if looks_dialogue and not has_domain and not has_url:
            cleaned.append(line)
            i += 1
            continue
        if promo_seed and (has_domain or has_url or (not long_narrative and len(normalized) <= max_len)):
            block_end = i + 1
            while block_end < len(lines):
                next_norm = normalize_line_for_filters(lines[block_end])
                if not next_norm:
                    block_end += 1
                    break
                next_lower = next_norm.lower()
                next_compact = re.sub(r"[^a-z0-9]", "", next_lower)
                next_domain = any(dom in next_lower for dom in domain_tokens) or any(tok in next_compact for tok in line_compact)
                next_url = bool(URL_RE.search(next_lower))
                next_phrase = any(phrase in next_lower for phrase in phrase_tokens)
                next_regex = any(r.search(lines[block_end]) for r in regexes)
                short_line = len(next_norm) <= 140
                letter_ratio = sum(1 for ch in next_norm if ch.isalpha()) / max(len(next_norm), 1)
                low_linguistic = letter_ratio < 0.65
                if next_domain or next_url or next_phrase or next_regex:
                    block_end += 1
                    continue
                if _is_dialogue_like(next_norm, lines[block_end]):
                    break
                if short_line and (low_linguistic or next_norm.endswith(":")):
                    block_end += 1
                    continue
                break
            removed_block = lines[i:block_end]
            if len(removed_block) > 1:
                stats["promo_blocks_removed_count"] += 1
            for removed in removed_block:
                removed_norm = normalize_line_for_filters(removed)
                removed_lower = removed_norm.lower()
                removed_compact = re.sub(r"[^a-z0-9]", "", removed_lower)
                removed_has_domain = any(dom in removed_lower for dom in line_contains) or any(
                    tok in removed_compact for tok in line_compact
                )
                removed_has_url = bool(URL_RE.search(removed_lower))
                _update_stats(removed_compact, has_url=removed_has_url, has_domain=removed_has_domain)
                _record_reason("block", removed_norm)
                if removed_norm:
                    removed_norms.append(removed_norm)
                    removed_lines.append(removed_norm)
            i = block_end
            continue

        if promo_seed and (len(normalized) <= max_len or has_url or has_domain):
            _update_stats(norm_compact, has_url=has_url, has_domain=has_domain)
            removed_norms.append(normalized)
            removed_lines.append(normalized)
            _record_reason("single", normalized)
            i += 1
            continue
        cleaned.append(line)
        i += 1

    remaining_counts = {
        "oceanofpdf": sum(1 for ln in cleaned if "oceanofpdf" in normalize_line_for_filters(ln).lower()),
        "gomanga": sum(1 for ln in cleaned if "gomanga.com" in normalize_line_for_filters(ln).lower()),
        "jnovels": sum(1 for ln in cleaned if "jnovels" in normalize_line_for_filters(ln).lower()),
        "zerobooks": sum(1 for ln in cleaned if "zerobooks" in normalize_line_for_filters(ln).lower()),
    }
    stats["remaining_counts"] = remaining_counts
    stats["promo_lines_removed_total"] = len(removed_norms)
    max_samples = 20
    stats["promo_samples"] = removed_lines[:max_samples]
    stats["promo_samples_truncated"] = len(removed_lines) > max_samples
    stats["promo_removed_hash"] = _sha256_text("\n".join(removed_lines)) if removed_lines else ""
    return "\n".join(cleaned), stats, removed_norms


def _dedupe_consecutive_lines(text: str) -> tuple[str, dict, list[str]]:
    lines = text.splitlines()
    cleaned: list[str] = []
    removed_norms: list[str] = []
    prev_norm: str | None = None
    removed_count = 0
    for line in lines:
        norm = normalize_line_for_filters(line)
        if norm and prev_norm and norm == prev_norm:
            removed_count += 1
            removed_norms.append(norm)
            continue
        cleaned.append(line)
        if norm:
            prev_norm = norm
        else:
            prev_norm = None
    return "\n".join(cleaned), {"dedupe_removed_count": removed_count}, removed_norms


def _fix_under_merge(text: str) -> tuple[str, dict]:
    lines = text.splitlines()
    fixed: list[str] = []
    merges = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        # remove isolated blank lines splitting mid-sentence when neighbors allow merge
        if not line.strip() and fixed:
            prev = fixed[-1] if fixed else ""
            if i + 1 < len(lines):
                nxt = lines[i + 1].lstrip()
                if prev and nxt and not re.search(r"[.!?…]['\"]?$", prev) and prev[-1].isalpha() and nxt[:1].islower():
                    fixed[-1] = f"{prev} {nxt}"
                    merges += 1
                    i += 2
                    continue
            fixed.append(line)
            i += 1
            continue
        if i + 1 < len(lines):
            curr = lines[i].rstrip()
            nxt = lines[i + 1].lstrip()
            if not curr or not nxt:
                fixed.append(lines[i])
                i += 1
                continue
            curr_is_heading = _is_heading_like(curr)
            ends_sentence = bool(re.search(r"[.!?…]['\"]?$", curr))
            starts_dialogue = nxt.startswith(('"', "“", "‘", "—", "-"))
            promo_guard = any(dom in nxt.lower() for dom in PROMO_DOMAINS) or URL_RE.search(nxt)
            next_is_heading = _is_heading_like(nxt)
            if curr.strip().isupper() and nxt.strip().isupper() and len(curr.strip()) <= 40 and len(nxt.strip()) <= 40:
                fixed.append(lines[i])
                i += 1
                continue
            if curr.strip().isupper() and len(curr.strip()) <= 40 and nxt[:1].isupper():
                fixed.append(lines[i])
                i += 1
                continue
            if (
                not ends_sentence
                and curr[-1].isalpha()
                and (nxt[:1].islower() or (nxt[:1].isupper() and not starts_dialogue and not next_is_heading))
                and not promo_guard
                and not curr_is_heading
            ):
                fixed.append(f"{curr} {nxt}")
                merges += 1
                i += 2
                continue
        fixed.append(lines[i])
        i += 1
    return "\n".join(fixed), {"under_merge_fixes": merges}


def _reflow_paragraphs(text: str) -> tuple[str, dict]:
    """
    Colapsa quebras duras dentro do mesmo par·grafo, mantendo linhas vazias e headings.

    HeurÌstica conservadora: n„o atravessa linhas vazias, headings ou separadores (***).
    """
    lines = text.splitlines()
    reflowed: list[str] = []
    buffer: list[str] = []
    merges = 0

    def _flush() -> None:
        nonlocal merges
        if not buffer:
            return
        if len(buffer) > 1:
            merges += len(buffer) - 1
        reflowed.append(" ".join(buffer))
        buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            _flush()
            reflowed.append("")
            continue
        if re.fullmatch(r"[.·…]{2,}", stripped):
            _flush()
            reflowed.append(stripped)
            continue
        if stripped.startswith(('"', "“")) and buffer:
            # nova fala: se anterior termina frase, quebra par·grafo; sen„o, continua frase
            if buffer[-1].rstrip().endswith((".", "!", "?", "…", '"', "”", "’")):
                _flush()
                buffer.append(stripped)
                continue
        if _is_heading_like(stripped) or re.fullmatch(r"\*{2,}", stripped):
            _flush()
            reflowed.append(line)
            continue
        if not buffer:
            buffer.append(line.strip())
        else:
            buffer.append(line.strip())
    _flush()
    return "\n".join(reflowed), {"reflow_merges": merges}


def _normalize_uppercase_sentences(text: str) -> tuple[str, dict]:
    """
    Converte linhas inteiras em CAPS (n„o headings) para sentence case seguro.
    Evita mexer em headings e separadores para n„o quebrar narrativa.
    """
    normalized: list[str] = []
    fixes = 0
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped
            and stripped.isupper()
            and len(stripped) > 8
            and " " in stripped
            and not _is_heading_like(stripped)
            and not re.fullmatch(r"\*{2,}", stripped)
        ):
            match = re.match(r"^([\"“‘(\\[]*)(.+)$", stripped)
            if match:
                prefix, body = match.groups()
                body_lower = body.lower()
                sentence = prefix + body_lower[:1].upper() + body_lower[1:]
                normalized.append(sentence)
                fixes += 1
            else:
                normalized.append(line)
        else:
            normalized.append(line)
    return "\n".join(normalized), {"uppercase_sentence_normalized": fixes}


def _strip_inline_watermarks(text: str) -> tuple[str, dict]:
    """
    Remove tokens de watermark/spam embutidos no meio de linhas (após merges).
    """
    patterns = [
        r"oceanofpdf\.com",
        r"zerobooks",
        r"jnovels",
        r"gomanga\.com",
        r"mp4directs\.com",
    ]
    combined = re.compile("|".join(patterns), flags=re.IGNORECASE)
    before = text
    text = combined.sub("", text)
    # normaliza espa‡os extras gerados pela remo‡Æo
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    removed = len(before) - len(text)
    return text, {"inline_watermark_removed_chars": removed}


def _fix_hyphen_linebreaks(text: str) -> tuple[str, dict]:
    pattern = re.compile(r"([A-Za-z]{1,24})-\s*\n\s*([A-Za-z]{1,24})")
    count = 0

    def _repl(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group(1)}-{match.group(2)}"

    fixed = pattern.sub(_repl, text)
    return fixed, {"hyphen_linebreak_fixes": count}


def _is_toc_entry(line: str) -> bool:
    stripped = normalize_line_for_filters(line)
    if not stripped:
        return False
    norm = stripped.lower()
    if any(marker in norm for marker in TOC_MARKER_LINES):
        return True
    if re.match(r"^(prologue|epilogue|afterword|chapter\s+\d+(:?[^\n]+)?)$", norm, flags=re.IGNORECASE):
        return True
    if re.match(r"^(pr[óo]logo|cap[ií]tulo\s+\d+|ep[ií]logo|p[oó]s-?escrito)", norm, flags=re.IGNORECASE):
        return True
    if len(stripped) < 80 and re.search(r"\b\d+\b", stripped):
        return True
    if len(stripped) < 120 and re.search(r"\.{2,}\s*\d{1,4}$", stripped):
        return True
    return False


def _is_heading_like(line: str) -> bool:
    norm = normalize_line_for_filters(line).lower()
    if not norm:
        return False
    if TOC_MARKER_RE.match(norm):
        return True
    if re.match(r"^(prologue|epilogue|afterword|chapter\s+\d+)", norm):
        return True
    if re.match(r"^(pr[óo]logo|cap[ií]tulo\s+\d+|ep[ií]logo)", norm):
        return True
    return False


def _remove_toc_blocks(
    text: str,
    *,
    head_window: int = 200,
    tail_window: int = 400,
    head_min_entries: int = 3,
    tail_min_entries: int = 12,
    min_density: float = 0.35,
    gap_limit: int = 8,
) -> tuple[str, dict]:
    lines = text.splitlines()
    removed_lines: list[str] = []
    head_removed = 0
    tail_removed = 0

    def _is_marker(norm: str) -> bool:
        if norm == "contents":
            return True
        return norm in TOC_MARKER_LINES

    def _looks_narrative(norm_line: str) -> bool:
        if not norm_line:
            return False
        if len(norm_line) >= 120:
            return True
        alpha = sum(1 for ch in norm_line if ch.isalpha())
        ratio = alpha / max(len(norm_line), 1)
        if len(norm_line) > 40 and ratio > 0.6 and bool(re.search(r"[.!?]", norm_line)):
            return True
        if len(norm_line) >= 20 and ratio > 0.7 and bool(re.search(r"[.!?]", norm_line)):
            return True
        return False

    def _strip_in_range(start: int, end: int, *, is_tail: bool = False) -> None:
        nonlocal head_removed, tail_removed, lines
        idx = start
        while idx < end and idx < len(lines):
            current_norm = normalize_line_for_filters(lines[idx]).lower()
            if _is_marker(current_norm):
                j = idx + 1
                limit = min(len(lines), idx + 200)
                toc_like = 1  # current line
                total = 1
                gap = 0
                last_idx = idx
                block_invalid = False
                while j < limit:
                    total += 1
                    candidate_norm = normalize_line_for_filters(lines[j]).lower()
                    if _is_toc_entry(candidate_norm) or re.search(r"\s\d{1,4}$", candidate_norm):
                        toc_like += 1
                        gap = 0
                        last_idx = j
                    else:
                        if _looks_narrative(candidate_norm):
                            block_invalid = True
                            break
                        gap += 1
                        if gap >= gap_limit:
                            break
                    j += 1
                density = toc_like / total if total else 0
                min_entries = tail_min_entries if is_tail else head_min_entries
                if block_invalid and not (toc_like >= min_entries and density >= min_density):
                    removed_lines.append(current_norm)
                    del lines[idx]
                    end = min(end, len(lines))
                    continue
                if toc_like >= min_entries and density >= min_density:
                    removed_lines.extend(normalize_line_for_filters(ln) for ln in lines[idx : last_idx + 1])
                    del lines[idx : last_idx + 1]
                    if is_tail:
                        tail_removed += 1
                    else:
                        head_removed += 1
                    end = min(end, len(lines))
                    continue
            idx += 1

    _strip_in_range(0, min(len(lines), head_window))
    tail_start = max(0, len(lines) - tail_window)
    _strip_in_range(tail_start, len(lines), is_tail=True)
    removed_hash = _sha256_text("\n".join(removed_lines)) if removed_lines else ""
    max_samples = 20
    removed_sample = removed_lines[:max_samples]
    stats = {
        "toc_blocks_removed_head": head_removed,
        "toc_blocks_removed_tail": tail_removed,
        "toc_blocks_removed_count": head_removed + tail_removed,
        "toc_lines_removed_count": len(removed_lines),
        "toc_removed_lines": removed_sample,
        "toc_removed_truncated": len(removed_lines) > max_samples,
        "toc_removed_hash": removed_hash,
    }
    return "\n".join(lines), stats


def _normalize_line_for_repeat(line: str) -> str:
    norm = normalize_line_for_filters(line).lower()
    norm = re.sub(r"\s+", " ", norm)
    norm = re.sub(r"[.,;:!?\-–—]+", "", norm)
    return norm


def _remove_repeated_lines(text: str, *, min_freq: int = 6, max_len: int = 80) -> tuple[str, dict, list[str]]:
    lines = text.splitlines()
    freq: Counter[str] = Counter()
    normalized: dict[int, str] = {}
    for idx, ln in enumerate(lines):
        norm = _normalize_line_for_repeat(ln)
        normalized[idx] = norm
        if norm:
            freq[norm] += 1

    to_remove: set[int] = set()
    for idx, norm in normalized.items():
        if not norm:
            continue
        if re.fullmatch(r"\*{2,}", lines[idx].strip()):
            continue
        count = freq.get(norm, 0)
        if count >= min_freq and (len(norm) <= max_len or any(dom in norm for dom in PROMO_DOMAINS)):
            to_remove.add(idx)

    cleaned = [ln for idx, ln in enumerate(lines) if idx not in to_remove]
    removed_norms = [normalized[idx] for idx in sorted(to_remove) if normalized.get(idx)]
    top_repeated = freq.most_common(10)
    stats = {
        "repeated_lines_removed_count": len(to_remove),
        "top_repeated_lines": top_repeated,
    }
    return "\n".join(cleaned), stats, removed_norms


def _fix_ocr_spacing(text: str) -> tuple[str, dict]:
    trailing_punct_re = re.compile("([-\u2013\u2014.,;:!?\u2026'\"”’]+)$")

    def _split_token(token: str) -> tuple[str, str]:
        match = trailing_punct_re.search(token)
        if not match:
            return token, ""
        start = match.start()
        return token[:start], token[start:]

    def _is_upperish(token: str) -> bool:
        core, _ = _split_token(token)
        clean_core = re.sub(r"[^A-Z]", "", core)
        return bool(clean_core) and bool(re.fullmatch(r"[A-Z]+", clean_core))

    def _fix_line(line: str) -> tuple[str, list[tuple[str, str]]]:
        tokens = line.split()
        new_tokens: list[str] = []
        samples: list[tuple[str, str]] = []
        i = 0
        while i < len(tokens):
            if _is_upperish(tokens[i]):
                seq: list[tuple[str, str, str, str]] = []
                long_count = 0
                while i < len(tokens) and _is_upperish(tokens[i]):
                    tok = tokens[i]
                    core, punct = _split_token(tok)
                    clean_core = re.sub(r"[^A-Z]", "", core)
                    if len(clean_core) > 1 and long_count >= 1 and len(seq) >= 2 and len(clean_core) > 6:
                        break
                    if len(clean_core) > 1:
                        long_count += 1
                    seq.append((tok, core, punct, clean_core))
                    i += 1
                cores = [core for (_, core, _, _) in seq]
                cores_clean = [core_clean for (_, _, _, core_clean) in seq]
                puncts = [punct for (_, _, punct, _) in seq]
                trail = next((p for p in reversed(puncts) if p), "")
                combined_core = "".join(cores)
                combined_clean = "".join(cores_clean)
                combined = combined_core + trail
                allow_merge = len(seq) >= 2 and long_count <= 1 and re.search(r"[AEIOU]", combined_clean)
                if len(seq) == 2 and cores_clean[0] == "I" and len(cores_clean[1]) > 3:
                    allow_merge = False
                if len(combined_clean) > 14:
                    allow_merge = False
                if len(seq) >= 3 and len(cores_clean[-1]) >= 8:
                    allow_merge = False
                if len(seq) >= 3 and len(cores_clean[-1]) >= 4:
                    allow_merge = False

                if allow_merge:
                    new_tokens.append(combined)
                    samples.append((" ".join(tok for (tok, _, _, _) in seq), combined))
                else:
                    if len(seq) >= 3 and all(len(c) == 1 for c in cores_clean[:-1]):
                        merged_prefix = "".join(cores_clean[:-1])
                        new_tokens.append(merged_prefix)
                        new_tokens.append(seq[-1][0])
                    elif (
                        len(seq) >= 2
                        and len(cores_clean[0]) == 1
                        and cores_clean[0] != "I"
                        and 2 <= len(cores_clean[1]) <= 12
                        and re.search(r"[AEIOU]", cores_clean[1])
                    ):
                        merged_prefix = cores[0] + cores[1] + (puncts[1] if len(puncts) > 1 else "")
                        new_tokens.append(merged_prefix)
                        if len(samples) < 10:
                            samples.append((f"{cores[0]} {cores[1]}", merged_prefix))
                        for tok, _, _, _ in seq[2:]:
                            new_tokens.append(tok)
                    else:
                        new_tokens.extend(tok for (tok, _, _, _) in seq)
            else:
                new_tokens.append(tokens[i])
                i += 1
        return " ".join(new_tokens), samples

    fixed_lines: list[str] = []
    samples: list[tuple[str, str]] = []
    total_fixes = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or not re.search(r"[A-Z]\s+[A-Z]", stripped):
            fixed_lines.append(line)
            continue
        fixed, local_samples = _fix_line(line)
        total_fixes += len(local_samples)
        if local_samples and len(samples) < 10:
            remaining = 10 - len(samples)
            samples.extend(local_samples[:remaining])
        fixed_lines.append(fixed)
    return "\n".join(fixed_lines), {"ocr_spacing_fixes": total_fixes, "ocr_spacing_samples": samples}


def _fix_spaced_caps_pairs(text: str) -> tuple[str, dict]:
    """
    Junta pares isolados de letras mai£sculas separados por espa‡o (ex.: W E -> WE).
    Conservador: exige vogal na palavra resultante ou tamanho pequeno.
    """
    pattern = re.compile(r"\b([A-Z])\s+([A-Z])\b")
    fixes = 0
    lines: list[str] = []
    for line in text.splitlines():
        new_line = line
        while True:
            match = pattern.search(new_line)
            if not match:
                break
            combined = f"{match.group(1)}{match.group(2)}"
            has_vowel = bool(re.search(r"[AEIOU]", combined))
            if has_vowel or len(combined) <= 3:
                new_line = new_line[: match.start()] + combined + new_line[match.end() :]
                fixes += 1
            else:
                break
        lines.append(new_line)
    return "\n".join(lines), {"spaced_caps_pair_fixes": fixes}


def strip_front_matter(text: str) -> str:
    """
    Remove tudo antes de Prologue/Chapter 1.
    """
    lines = text.splitlines()
    start_idx = None
    marker_re = re.compile(r"^(prologue|chapter\s*1:?|cap[ií]tulo\s*1:?|ep[ií]logo)", re.IGNORECASE)
    for idx, ln in enumerate(lines):
        if marker_re.match(ln.strip()):
            start_idx = idx
            break
    if start_idx is None:
        return text
    return "\n".join(lines[start_idx:]).lstrip()


def strip_toc(
    text: str,
    logger: Optional[logging.Logger] = None,
    *,
    max_lines: int = 200,
    min_markers: int = 4,
    max_body_len: int = 50,
) -> str:
    """
    Remove sumário/TOC no início detectando marcadores curtos em sequência.
    Heurística: >= min_markers nos primeiros max_lines e maioria dos corpos entre eles vazios/curtos.
    """
    lines = text.splitlines()
    search_lines = lines[:max_lines]
    marker_idxs: list[int] = []
    for idx, raw in enumerate(search_lines):
        normalized = raw.strip()
        if normalized.startswith("#"):
            normalized = normalized.lstrip("#").strip()
        if TOC_MARKER_RE.match(normalized):
            marker_idxs.append(idx)

    if len(marker_idxs) < min_markers:
        return text

    short_bodies = 0
    total_bodies = 0
    for i, start_idx in enumerate(marker_idxs):
        end_idx = marker_idxs[i + 1] if i + 1 < len(marker_idxs) else len(search_lines)
        body_text = "\n".join(search_lines[start_idx + 1 : end_idx]).strip()
        total_bodies += 1
        if len(body_text) < max_body_len:
            short_bodies += 1

    if not total_bodies or short_bodies / total_bodies < 0.6:
        return text

    cutoff_line = marker_idxs[-1] + 1
    # Avan‡a al‚m de linhas vazias ou num‚ricas logo ap¢s o £ltimo marcador (ex.: n£mero de p gina do sum rio).
    while cutoff_line < len(lines):
        candidate = lines[cutoff_line].strip()
        if not candidate:
            cutoff_line += 1
            continue
        if len(candidate) < max_body_len and not re.search(r"[A-Za-z]", candidate):
            cutoff_line += 1
            continue
        break
    remaining = lines[cutoff_line:]
    cleaned = "\n".join(remaining)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).lstrip()
    if logger:
        logger.info(
            "strip_toc: removido TOC inicial (markers=%d, short_bodies=%d/%d, cutoff_line=%d)",
            len(marker_idxs),
            short_bodies,
            total_bodies,
            cutoff_line,
        )
    return cleaned


def preprocess_text(
    raw_text: str,
    logger: Optional[logging.Logger] = None,
    *,
    skip_front_matter: bool = False,
    return_stats: bool = False,
    noise_glossary_path: str | Path | None = None,
) -> str | tuple[str, dict]:
    """
    Pré-processa o texto bruto extraído do PDF:
    - Normaliza quebras de linha
    - Remove rodapés/watermarks e blocos de ruído (ads/newsletter/discord)
    - Remove front-matter/TOC quando skip_front_matter=True
    """

    stats: dict[str, int | dict] = {"chars_in": len(raw_text)}
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    stats["soft_hyphen_removed"] = text.count("\u00ad")
    text = text.replace("\u00ad", "")
    text = ZERO_WIDTH_RE.sub("", text.replace("\xa0", " "))
    text, sanitize_stats = sanitize_extracted_text(text, logger=logger)
    stats.update(sanitize_stats)
    glossary = _load_noise_glossary(noise_glossary_path)
    if skip_front_matter:
        text = strip_front_matter(text)
        text = strip_toc(text, logger=logger)

    for pattern in FOOTER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    removed_counter: Counter[str] = Counter()

    text, promo_stats, promo_removed = _remove_promo_lines(text, glossary)
    stats.update(promo_stats)
    removed_counter.update(promo_removed)

    text, toc_stats = _remove_toc_blocks(text)
    stats.update(toc_stats)
    removed_counter.update(toc_stats.get("toc_removed_lines", []))

    text, hyphen_stats = _fix_hyphen_linebreaks(text)
    stats.update(hyphen_stats)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()
    text, ocr_stats = _fix_ocr_spacing(text)
    stats.update(ocr_stats)
    text, spaced_pair_stats = _fix_spaced_caps_pairs(text)
    stats.update(spaced_pair_stats)

    text, under_merge_stats = _fix_under_merge(text)
    stats.update(under_merge_stats)

    text = remove_noise_blocks(text)
    text, repeat_stats, repeat_removed = _remove_repeated_lines(text)
    stats.update(repeat_stats)
    removed_counter.update(repeat_removed)

    text, dedupe_stats, dedupe_removed = _dedupe_consecutive_lines(text)
    stats.update(dedupe_stats)
    removed_counter.update(dedupe_removed)

    text, reflow_stats = _reflow_paragraphs(text)
    stats.update(reflow_stats)

    text, upper_stats = _normalize_uppercase_sentences(text)
    stats.update(upper_stats)

    text, inline_stats = _strip_inline_watermarks(text)
    stats.update(inline_stats)

    # Restaura heading de pr¢logo se ele existia no raw mas nÆo sobrou ap¢s a limpeza.
    if (
        not skip_front_matter
        and "prologue" in normalize_line_for_filters(raw_text).lower()
        and "prologue" not in normalize_line_for_filters(text).lower()
    ):
        lines = text.splitlines()
        insert_at = 0
        for idx, ln in enumerate(lines):
            if ln.strip():
                insert_at = idx
                break
        lines.insert(insert_at, "Prologue")
        text = "\n".join(lines)

    # Valida‡Æo leve de sa¡da
    watermarks_remaining = sum(
        1
        for ln in text.splitlines()
        if any(tok in normalize_line_for_filters(ln).lower() for tok in ("oceanofpdf", "zerobooks", "jnovels", "gomanga.com", "newsletter"))
    )
    stats["watermarks_remaining"] = watermarks_remaining
    stats["soft_hyphen_remaining"] = text.count("\u00ad")
    stats["spaced_caps_remaining"] = len(re.findall(r"\b[A-Z]\s+[A-Z]{2,}\b", text))
    # primeira linha plausÌvel
    first_non_empty = next((ln for ln in text.splitlines() if ln.strip()), "")
    stats["first_line"] = first_non_empty

    stats["removed_lines_total"] = sum(removed_counter.values())
    stats["removed_lines_top"] = removed_counter.most_common(10)
    stats["chars_out"] = len(text)

    if logger is not None:
        logger.debug("Texto pré-processado: %d caracteres", len(text))

    if return_stats:
        return text, stats
    return text


def paragraphs_from_text(clean_text: str) -> List[str]:
    """Divide texto limpo em parágrafos usando quebras duplas."""
    return [p.strip() for p in clean_text.split("\n\n") if p.strip()]


def chunk_for_translation(paragraphs: List[str], max_chars: int, logger: logging.Logger) -> List[str]:
    """
    Chunk seguro para tradução com ajuste leve por fronteira de frase.

    Usa max_chars como alvo, mas permite pequeno lookahead para fechar
    o chunk no fim de frase (., ?, !) evitando cortar falas.
    """
    text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    if not text:
        return []

    boundary_re = re.compile(r"\n\n|[.!?](?:['\"”])?(?=\s|\n|$)")
    chunks: List[str] = []
    start = 0
    total_len = len(text)
    lookahead = 400  # permite estouro controlado para terminar frase
    consumed = 0

    while start < total_len:
        target_end = start + max_chars
        hard_end = min(total_len, start + max_chars + lookahead)

        if target_end >= total_len:
            last_slice = text[start:]
            chunks.append(last_slice.strip())
            consumed += len(last_slice)
            break

        window = text[start:hard_end]
        after_target: int | None = None
        before_target: int | None = None

        for match in boundary_re.finditer(window):
            end_pos = start + match.end()
            if target_end <= end_pos <= hard_end:
                after_target = end_pos
            elif end_pos < target_end:
                before_target = end_pos

        if after_target:
            chunk_end = after_target
            logger.debug(
                "tradução: chunk fechado em fim de frase após lookahead (len=%d)",
                chunk_end - start,
            )
        elif before_target:
            chunk_end = before_target
            logger.debug(
                "tradução: chunk fechado em limite seguro antes do alvo (len=%d)",
                chunk_end - start,
            )
        else:
            chunk_end = min(target_end, total_len)
            logger.debug("tradução: chunk fechado no alvo (len=%d)", chunk_end - start)

        if chunk_end <= start:
            chunk_end = min(target_end, total_len)

        raw_slice = text[start:chunk_end]
        chunks.append(raw_slice.strip())
        consumed += len(raw_slice)
        start = chunk_end

    if consumed != total_len:
        logger.warning("tradução: soma dos chunks (%d) difere do texto original (%d)", consumed, total_len)

    return chunks


def chunk_for_translation_with_offsets(
    paragraphs: List[str],
    max_chars: int,
    logger: logging.Logger,
) -> List[tuple[str, int | None, int | None]]:
    """
    Chunk seguro para tradução com offsets (start/end) no texto-base.

    Mantém a mesma lógica de `chunk_for_translation`, retornando tuplas
    (chunk_text, start_offset, end_offset).
    """
    text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    if not text:
        return []

    boundary_re = re.compile(r"\n\n|[.!?](?:['\"”])?(?=\s|\n|$)")
    chunks: List[tuple[str, int | None, int | None]] = []
    start = 0
    total_len = len(text)
    lookahead = 400
    consumed = 0

    while start < total_len:
        target_end = start + max_chars
        hard_end = min(total_len, start + max_chars + lookahead)

        if target_end >= total_len:
            raw_slice = text[start:]
            chunk_text = raw_slice.strip()
            leading = len(raw_slice) - len(raw_slice.lstrip())
            trailing = len(raw_slice) - len(raw_slice.rstrip())
            start_offset = start + leading if chunk_text else None
            end_offset = (start + len(raw_slice) - trailing) if chunk_text else None
            chunks.append((chunk_text, start_offset, end_offset))
            consumed += len(raw_slice)
            break

        window = text[start:hard_end]
        after_target: int | None = None
        before_target: int | None = None

        for match in boundary_re.finditer(window):
            end_pos = start + match.end()
            if target_end <= end_pos <= hard_end:
                after_target = end_pos
            elif end_pos < target_end:
                before_target = end_pos

        if after_target:
            chunk_end = after_target
            logger.debug(
                "tradução: chunk fechado em fim de frase após lookahead (len=%d)",
                chunk_end - start,
            )
        elif before_target:
            chunk_end = before_target
            logger.debug(
                "tradução: chunk fechado em limite seguro antes do alvo (len=%d)",
                chunk_end - start,
            )
        else:
            chunk_end = min(target_end, total_len)
            logger.debug("tradução: chunk fechado no alvo (len=%d)", chunk_end - start)

        if chunk_end <= start:
            chunk_end = min(target_end, total_len)

        raw_slice = text[start:chunk_end]
        chunk_text = raw_slice.strip()
        leading = len(raw_slice) - len(raw_slice.lstrip())
        trailing = len(raw_slice) - len(raw_slice.rstrip())
        start_offset = start + leading if chunk_text else None
        end_offset = (start + len(raw_slice) - trailing) if chunk_text else None
        chunks.append((chunk_text, start_offset, end_offset))
        consumed += len(raw_slice)
        start = chunk_end

    if consumed != total_len:
        logger.warning("tradução: soma dos chunks (%d) difere do texto original (%d)", consumed, total_len)

    return chunks


def chunk_for_refine(paragraphs: List[str], max_chars: int, logger: logging.Logger) -> List[str]:
    """Chunk seguro para refine com limite estrito."""
    return chunk_by_paragraphs(paragraphs, max_chars=max_chars, logger=logger, label="refine")
