from tradutor.structure_normalizer import normalize_structure


def test_heading_with_inline_text_is_split():
    text = "Prólogo SOGOU AYAKA PAROU O GOLPE.\n\nOutro parágrafo."
    result = normalize_structure(text)
    assert "Prólogo" in result
    lines = [ln for ln in result.splitlines() if ln.strip()]
    assert lines[0] == "Prólogo"
    assert lines[1] == "SOGOU AYAKA PAROU O GOLPE."


def test_idempotent_normalize_structure():
    text = "Prólogo\n\nSOGOU AYAKA PAROU O GOLPE.\n\nOutro parágrafo."
    once = normalize_structure(text)
    twice = normalize_structure(once)
    assert once == twice
