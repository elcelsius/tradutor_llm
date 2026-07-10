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


def test_chapter_title_glued_to_first_sentence_is_split():
    text = "# Capítulo 1:\n\nDepois do Deathmatch DEPOIS QUE SOGOU viu o corpo do Kirihara congelar."
    result = normalize_structure(text)

    assert result.startswith("# Capítulo 1: Após o Combate Mortal")
    assert "Depois que Sogou viu o corpo do Kirihara congelar." in result
    assert "Deathmatch DEPOIS" not in result


def test_missing_chapter_heading_for_deathmatch_opening_is_restored():
    text = "Após o combate mortal, após Sogou ter visto o corpo do Kirihara congelar, ela perdeu os sentidos."
    result = normalize_structure(text)

    assert result.startswith("# Capítulo 1: Após o Combate Mortal")
    assert "Após Sogou ter visto o corpo do Kirihara congelar" in result
