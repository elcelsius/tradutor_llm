"""
Pré-processamento de PDFs: extração de texto, limpeza e chunking seguro.
"""

from __future__ import annotations

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
    "contents",
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


def _remove_promo_lines(text: str) -> tuple[str, dict, list[str]]:
    lines = text.splitlines()
    cleaned: list[str] = []
    stats = {
        "oceanofpdf_removed_count": 0,
        "promo_lines_removed_count": 0,
        "urls_removed_count": 0,
    }
    removed_norms: list[str] = []
    compact_tokens = [re.sub(r"[^a-z0-9]", "", dom) for dom in PROMO_DOMAINS]
    for line in lines:
        normalized = normalize_line_for_filters(line)
        norm_lower = normalized.lower()
        norm_compact = re.sub(r"[^a-z0-9]", "", norm_lower)
        if not normalized:
            cleaned.append("")
            continue
        is_short = len(normalized) <= 160
        has_domain = any(dom in norm_lower for dom in PROMO_DOMAINS) or any(tok in norm_compact for tok in compact_tokens)
        has_url = bool(URL_RE.search(norm_lower))
        has_phrase = any(phrase in norm_lower for phrase in PROMO_PHRASES)
        looks_dialogue = line.lstrip().startswith(("\"", "“", "’", "‘", "—", "-"))
        if looks_dialogue and not has_domain and not has_url:
            cleaned.append(line)
            continue
        long_narrative = len(normalized) > 180 and not (has_url or has_domain)
        if "oceanofpdf" in norm_compact and (is_short or has_domain or has_url or has_phrase or long_narrative):
            stats["oceanofpdf_removed_count"] += 1
            stats["promo_lines_removed_count"] += 1
            if has_url:
                stats["urls_removed_count"] += 1
            removed_norms.append(normalized)
            continue
        if (has_domain or has_url or has_phrase) and not long_narrative:
            stats["promo_lines_removed_count"] += 1
            if has_domain or has_url:
                stats["urls_removed_count"] += 1
            if "oceanofpdf" in norm_compact:
                stats["oceanofpdf_removed_count"] += 1
            removed_norms.append(normalized)
            continue
        cleaned.append(line)
    remaining_counts = {
        "oceanofpdf": sum(1 for ln in cleaned if "oceanofpdf" in normalize_line_for_filters(ln).lower()),
        "gomanga": sum(1 for ln in cleaned if "gomanga.com" in normalize_line_for_filters(ln).lower()),
        "jnovels": sum(1 for ln in cleaned if "jnovels" in normalize_line_for_filters(ln).lower()),
        "zerobooks": sum(1 for ln in cleaned if "zerobooks" in normalize_line_for_filters(ln).lower()),
    }
    stats["remaining_counts"] = remaining_counts
    stats["promo_lines_removed_total"] = len(removed_norms)
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
            ends_sentence = bool(re.search(r"[.!?…]['\"]?$", curr))
            starts_dialogue = nxt.startswith(('"', "“", "‘", "—", "-"))
            promo_guard = any(dom in nxt.lower() for dom in PROMO_DOMAINS) or URL_RE.search(nxt)
            if (
                not ends_sentence
                and not starts_dialogue
                and curr[-1].isalpha()
                and nxt[:1].islower()
                and not promo_guard
            ):
                fixed.append(f"{curr} {nxt}")
                merges += 1
                i += 2
                continue
        fixed.append(lines[i])
        i += 1
    return "\n".join(fixed), {"under_merge_fixes": merges}


def _fix_hyphen_linebreaks(text: str) -> tuple[str, dict]:
    pattern = re.compile(r"([A-Za-z]{2,20})-\s*\n\s*([a-z]{1,20})")
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
        return True
    norm = stripped.lower()
    if any(marker in norm for marker in TOC_MARKER_LINES):
        return True
    if re.match(r"^(prologue|epilogue|afterword|chapter\s+\d+(:?[^\n]+)?)$", norm, flags=re.IGNORECASE):
        return True
    if re.match(r"^(pr[óo]logo|cap[ií]tulo\s+\d+|ep[ií]logo|p[oó]s-?escrito)", norm, flags=re.IGNORECASE):
        return True
    if len(stripped) < 80 and re.search(r"\b\d+\b", stripped):
        return True
    if len(stripped) < 80 and stripped.count(".") >= 2:
        return True
    return False


def _remove_toc_blocks(
    text: str,
    *,
    head_window: int = 200,
    tail_window: int = 400,
    head_min_entries: int = 4,
    tail_min_entries: int = 12,
    min_density: float = 0.35,
    gap_limit: int = 8,
) -> tuple[str, dict]:
    lines = text.splitlines()
    removed_lines: list[str] = []
    head_removed = 0
    tail_removed = 0

    def _is_marker(norm: str) -> bool:
        return norm in TOC_MARKER_LINES or any(marker in norm for marker in TOC_MARKER_LINES)

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
                while j < limit:
                    total += 1
                    candidate_norm = normalize_line_for_filters(lines[j]).lower()
                    if _is_toc_entry(candidate_norm) or re.search(r"\s\d{1,4}$", candidate_norm):
                        toc_like += 1
                        gap = 0
                        last_idx = j
                    else:
                        gap += 1
                        if gap >= gap_limit:
                            break
                    j += 1
                density = toc_like / total if total else 0
                min_entries = tail_min_entries if is_tail else head_min_entries
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
    stats = {
        "toc_blocks_removed_head": head_removed,
        "toc_blocks_removed_tail": tail_removed,
        "toc_blocks_removed_count": head_removed + tail_removed,
        "toc_lines_removed_count": len(removed_lines),
        "toc_removed_lines": removed_lines,
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
    def _split_token(token: str) -> tuple[str, str]:
        match = re.match(r"([A-Z’']+)([—\-.,;:!?]*)$", token)
        if match:
            return match.group(1), match.group(2)
        return token, ""

    def _is_upperish(token: str) -> bool:
        core, _ = _split_token(token)
        return bool(re.fullmatch(r"[A-Z](?:[’']?[A-Z]+)?", core))

    def _fix_line(line: str) -> tuple[str, list[tuple[str, str]]]:
        tokens = line.split()
        new_tokens: list[str] = []
        samples: list[tuple[str, str]] = []
        i = 0
        while i < len(tokens):
            if _is_upperish(tokens[i]):
                seq: list[str] = []
                long_count = 0
                while i < len(tokens) and _is_upperish(tokens[i]):
                    tok = tokens[i]
                    if len(tok) > 1 and long_count >= 1 and len(seq) >= 2 and len(tok) > 6:
                        break
                    if len(tok) > 1:
                        long_count += 1
                    seq.append(tok)
                    i += 1
                cores = []
                trail = ""
                for part in seq:
                    core, punct = _split_token(part)
                    cores.append(core)
                    trail = punct or trail
                combined_core = "".join(cores)
                combined = combined_core + trail
                allow_merge = len(seq) >= 2 and long_count <= 1 and re.search(r"[AEIOU]", combined_core)
                if len(seq) == 2 and cores[0] == "I" and len(cores[1]) > 3:
                    allow_merge = False
                if len(combined_core) > 9:
                    allow_merge = False
                if len(seq) >= 3 and len(cores[-1]) >= 8:
                    allow_merge = False
                if len(seq) >= 3 and len(cores[-1]) >= 4:
                    allow_merge = False

                if allow_merge:
                    new_tokens.append(combined)
                    samples.append((" ".join(seq), combined))
                else:
                    # try partial merge for short prefixes (e.g., W E CONTINUED -> WE CONTINUED)
                    if len(seq) >= 3 and all(len(c) == 1 for c in cores[:-1]):
                        merged_prefix = "".join(cores[:-1])
                        new_tokens.append(merged_prefix)
                        new_tokens.append(seq[-1])
                    elif len(seq) >= 2 and len(cores[0]) == 1 and len(cores[1]) <= 4:
                        merged_prefix = "".join(cores[:2])
                        new_tokens.append(merged_prefix)
                        new_tokens.extend(seq[2:])
                    else:
                        new_tokens.extend(seq)
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
) -> str | tuple[str, dict]:
    """
    Pré-processa o texto bruto extraído do PDF:
    - Normaliza quebras de linha
    - Remove rodapés/watermarks e blocos de ruído (ads/newsletter/discord)
    - Remove front-matter/TOC quando skip_front_matter=True
    """

    stats: dict[str, int | dict] = {"chars_in": len(raw_text)}
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = ZERO_WIDTH_RE.sub("", text.replace("\xa0", " "))
    text, sanitize_stats = sanitize_extracted_text(text, logger=logger)
    stats.update(sanitize_stats)
    if skip_front_matter:
        text = strip_front_matter(text)
        text = strip_toc(text, logger=logger)

    for pattern in FOOTER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    removed_counter: Counter[str] = Counter()

    text, promo_stats, promo_removed = _remove_promo_lines(text)
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

    text, under_merge_stats = _fix_under_merge(text)
    stats.update(under_merge_stats)

    text = remove_noise_blocks(text)
    text, repeat_stats, repeat_removed = _remove_repeated_lines(text)
    stats.update(repeat_stats)
    removed_counter.update(repeat_removed)

    text, dedupe_stats, dedupe_removed = _dedupe_consecutive_lines(text)
    stats.update(dedupe_stats)
    removed_counter.update(dedupe_removed)

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
