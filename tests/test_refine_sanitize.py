from tradutor.refine import sanitize_refine_chunk_output


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
