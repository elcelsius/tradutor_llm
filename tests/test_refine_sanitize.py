from tradutor.refine import sanitize_refine_chunk_output
from tradutor.sanitizer import sanitize_refine_output


def test_sanitize_removes_trailing_triple_quotes():
    raw = 'Texto suficiente."""\nLinha ok.'
    cleaned, ok, info = sanitize_refine_chunk_output(raw, raw, logger=None, label="t1")
    assert ok
    assert '"""' not in cleaned
    assert cleaned.startswith("Texto suficiente.")
    assert info["blank_lines_fixed"] == 0


def test_sanitize_collapses_blank_lines_inside_quotes():
    raw = "“Entendo.\n\nQuer dizer...”\n\nFora do dialogo."
    cleaned, ok, _ = sanitize_refine_chunk_output(raw, raw, logger=None, label="t2")
    assert ok
    assert "Entendo.\nQuer dizer" in cleaned
    # Paragrafo fora das aspas permanece
    assert "\n\nFora do dialogo." in cleaned


def test_sanitize_splits_glued_dialogues():
    raw = "“Oi.” “Tchau.”"
    cleaned, ok, info = sanitize_refine_chunk_output(raw, raw, logger=None, label="t3")
    assert ok
    assert "”\n\n“" in cleaned
    assert info["dialogue_splits"] == 1


def test_sanitize_keeps_dialogue_tag_attached():
    raw = "“Oi.”\n\nperguntou Marla."
    cleaned, ok, _ = sanitize_refine_chunk_output(raw, raw, logger=None, label="t4")
    assert ok
    assert "” perguntou Marla." in cleaned
    assert "\n\nperguntou" not in cleaned


def test_sanitize_rejects_quote_to_dash_dialogue_conversion():
    original = '"Oi", disse Marla.\n\n"Sim", respondeu ele.'
    raw = '— Oi", disse Marla.\n\n— Sim", respondeu ele.'
    _, ok, info = sanitize_refine_chunk_output(raw, original, logger=None, label="t5")
    assert not ok
    assert info["dialogue_style_changed"]


def test_sanitize_allows_existing_dash_dialogues():
    original = "— Oi, disse Marla.\n\n— Sim, respondeu ele."
    raw = "— Oi, disse Marla.\n\n— Sim, respondeu ele."
    _, ok, info = sanitize_refine_chunk_output(raw, original, logger=None, label="t6")
    assert ok
    assert not info["dialogue_style_changed"]


def test_sanitize_rejects_large_paragraph_reflow():
    original = "Um.\n\nDois.\n\nTres.\n\nQuatro."
    raw = "Um.\n\nDois.\n\nTres.\n\nQuatro.\n\nCinco.\n\nSeis.\n\nSete."
    _, ok, info = sanitize_refine_chunk_output(raw, original, logger=None, label="t7")
    assert not ok
    assert info["paragraph_structure_changed"]


def test_sanitize_rejects_large_line_reflow_inside_paragraphs():
    original = "Um. Dois. Tres. Quatro.\n\nCinco. Seis. Sete. Oito."
    raw = "Um.\nDois.\nTres.\nQuatro.\n\nCinco.\nSeis.\nSete.\nOito."
    _, ok, info = sanitize_refine_chunk_output(raw, original, logger=None, label="t8")
    assert not ok
    assert info["line_structure_changed"]


def test_sanitize_refine_output_removes_refined_markers_only():
    raw = "# Titulo\n\n### TEXTO_REFINADO_INICIO\n\nTexto refinado.\n\n### TEXTO_REFINADO_FIM"

    cleaned = sanitize_refine_output(raw)

    assert "TEXTO_REFINADO" not in cleaned
    assert "# Titulo" in cleaned
    assert "Texto refinado." in cleaned
