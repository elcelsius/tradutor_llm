from tradutor.refine import sanitize_refine_chunk_output
from tradutor.sanitizer import sanitize_refine_output


def test_sanitize_removes_trailing_triple_quotes():
    """Processamento interno auxiliar."""
    raw = 'Texto suficiente."""\nLinha ok.'
    cleaned, ok, info = sanitize_refine_chunk_output(raw, raw, logger=None, label="t1")
    assert ok
    assert '"""' not in cleaned
    assert cleaned.startswith("Texto suficiente.")
    assert info["blank_lines_fixed"] == 0


def test_sanitize_collapses_blank_lines_inside_quotes():
    """Processamento interno auxiliar."""
    raw = "“Entendo.\n\nQuer dizer...”\n\nFora do dialogo."
    cleaned, ok, _ = sanitize_refine_chunk_output(raw, raw, logger=None, label="t2")
    assert ok
    assert "Entendo. Quer dizer" in cleaned
    # Paragrafo fora das aspas permanece
    assert "\n\nFora do dialogo." in cleaned


def test_sanitize_joins_standalone_closing_quote_after_blank_line() -> None:
    """Processamento interno auxiliar."""
    raw = "“Entendo?\n\n”\n\nFora do dialogo."

    cleaned, ok, _ = sanitize_refine_chunk_output(raw, raw, logger=None, label="t2b")

    assert ok
    assert "“Entendo?”" in cleaned
    assert "\n\n”" not in cleaned


def test_sanitize_splits_glued_dialogues():
    """Processamento interno auxiliar."""
    raw = "“Oi.” “Tchau.”"
    cleaned, ok, info = sanitize_refine_chunk_output(raw, raw, logger=None, label="t3")
    assert ok
    assert "”\n\n“" in cleaned
    assert info["dialogue_splits"] == 1


def test_sanitize_keeps_dialogue_tag_attached():
    """Processamento interno auxiliar."""
    raw = "“Oi.”\n\nperguntou Marla."
    cleaned, ok, _ = sanitize_refine_chunk_output(raw, raw, logger=None, label="t4")
    assert ok
    assert "” perguntou Marla." in cleaned
    assert "\n\nperguntou" not in cleaned


def test_sanitize_rejects_quote_to_dash_dialogue_conversion():
    """Processamento interno auxiliar."""
    original = '"Oi", disse Marla.\n\n"Sim", respondeu ele.'
    raw = '— Oi", disse Marla.\n\n— Sim", respondeu ele.'
    _, ok, info = sanitize_refine_chunk_output(raw, original, logger=None, label="t5")
    assert not ok
    assert info["dialogue_style_changed"]


def test_sanitize_allows_existing_dash_dialogues():
    """Processamento interno auxiliar."""
    original = "— Oi, disse Marla.\n\n— Sim, respondeu ele."
    raw = "— Oi, disse Marla.\n\n— Sim, respondeu ele."
    _, ok, info = sanitize_refine_chunk_output(raw, original, logger=None, label="t6")
    assert ok
    assert not info["dialogue_style_changed"]


def test_sanitize_rejects_large_paragraph_reflow():
    """Processamento interno auxiliar."""
    original = "Um.\n\nDois.\n\nTres.\n\nQuatro."
    raw = "Um.\n\nDois.\n\nTres.\n\nQuatro.\n\nCinco.\n\nSeis.\n\nSete."
    _, ok, info = sanitize_refine_chunk_output(raw, original, logger=None, label="t7")
    assert not ok
    assert info["paragraph_structure_changed"]


def test_sanitize_rejects_single_paragraph_split_in_refine_chunk():
    """Processamento interno auxiliar."""
    original = "Um.\n\nDois.\n\nTres.\n\nQuatro."
    raw = "Um.\n\nDois.\n\nTres.\n\nQuatro.\n\nCinco."
    _, ok, info = sanitize_refine_chunk_output(raw, original, logger=None, label="t7b")
    assert not ok
    assert info["paragraph_structure_changed"]


def test_sanitize_allows_own_glued_dialogue_split():
    """Processamento interno auxiliar."""
    original = "Um.\n\nDois.\n\nTres.\n\n“Oi.” “Tchau.”"
    raw = original
    cleaned, ok, info = sanitize_refine_chunk_output(
        raw, original, logger=None, label="t7c"
    )
    assert ok
    assert "”\n\n“" in cleaned
    assert info["dialogue_splits"] == 1
    assert not info["paragraph_structure_changed"]


def test_sanitize_rejects_large_line_reflow_inside_paragraphs():
    """Processamento interno auxiliar."""
    original = "Um. Dois. Tres. Quatro.\n\nCinco. Seis. Sete. Oito."
    raw = "Um.\nDois.\nTres.\nQuatro.\n\nCinco.\nSeis.\nSete.\nOito."
    _, ok, info = sanitize_refine_chunk_output(raw, original, logger=None, label="t8")
    assert not ok
    assert info["line_structure_changed"]


def test_sanitize_refine_output_removes_refined_markers_only():
    """Processamento interno auxiliar."""
    raw = "# Titulo\n\n### TEXTO_REFINADO_INICIO\n\nTexto refinado.\n\n### TEXTO_REFINADO_FIM"

    cleaned = sanitize_refine_output(raw)

    assert "TEXTO_REFINADO" not in cleaned
    assert "# Titulo" in cleaned
    assert "Texto refinado." in cleaned


def test_sanitize_refine_output_extracts_delimited_body_after_meta_intro():
    """Processamento interno auxiliar."""
    raw = (
        "Aqui está a revisão do texto:\n\n***\n\n"
        "Texto revisado, sem metacomentários.\n\n***\n\n"
        "### Principais ajustes feitos:\n- Ajuste de fluidez."
    )

    assert sanitize_refine_output(raw) == "Texto revisado, sem metacomentários."


def test_sanitize_refine_output_preserves_scene_separators_without_meta_intro():
    """Processamento interno auxiliar."""
    raw = "Cena um.\n\n***\n\nCena dois.\n\n***\n\nCena três."

    assert sanitize_refine_output(raw) == raw


def test_sanitize_refine_output_extracts_named_review_body_and_drops_notes():
    """Processamento interno auxiliar."""
    raw = (
        "Aqui está a revisão:\n\n**Texto Revisado:**\n\n"
        "Texto da novel.\n\n---\n\n### Principais Ajustes Feitos:\n- Nota."
    )

    assert sanitize_refine_output(raw) == "Texto da novel."


def test_sanitize_requests_retry_for_new_malformed_quote_boundary():
    """Processamento interno auxiliar."""
    original = "“Ah, tudo bem.”"
    raw = "”“Ah, tudo bem.”"

    _, ok, info = sanitize_refine_chunk_output(
        raw, original, logger=None, label="quote-boundary"
    )

    assert ok
    assert info["soft_retry"]
    assert info["introduced_malformed_quote_boundary"]


def test_sanitize_requests_retry_for_extra_balanced_quote_pair() -> None:
    """Processamento interno auxiliar."""
    original = "“Primeira fala.”"
    raw = "“Primeira fala.”\n\n“Fala inventada.”"

    _, ok, info = sanitize_refine_chunk_output(
        raw, original, logger=None, label="quote-count"
    )

    assert ok
    assert info["soft_retry"]
    assert info["introduced_extra_curly_quotes"]
