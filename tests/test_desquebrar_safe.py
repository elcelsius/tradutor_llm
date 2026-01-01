from tradutor.desquebrar_safe import safe_reflow, desquebrar_safe


def test_join_simple_continuation():
    raw = "Primeira parte\ncontinua aqui."
    assert safe_reflow(raw) == "Primeira parte continua aqui."


def test_join_hyphenation_without_space():
    raw = "palavra-\nquebrada no meio"
    assert safe_reflow(raw) == "palavraquebrada no meio"


def test_short_line_not_treated_as_title():
    raw = "Em uma linha\ncontinua sem titulo"
    assert safe_reflow(raw) == "Em uma linha continua sem titulo"


def test_preserve_dialogue_and_blank_lines():
    raw = '"Oi."\nela respondeu.\n\n"Nova fala"\nsegue aqui'
    expected = '"Oi."\nela respondeu.\n\n"Nova fala"\nsegue aqui'
    assert safe_reflow(raw) == expected


def test_preserve_em_dash_dialogue_start():
    raw = "Linha anterior\n— Então comecou.\ncontinua aqui"
    expected = "Linha anterior\n— Então comecou.\ncontinua aqui"
    assert safe_reflow(raw) == expected


def test_block_join_when_next_is_uppercase_or_title():
    raw = "final de frase\nProximo Paragrafo\nCAPITULO UM\ntexto inicia"
    expected = "final de frase\nProximo Paragrafo\nCAPITULO UM\ntexto inicia"
    assert safe_reflow(raw) == expected


def test_desquebrar_safe_wrapper():
    raw = "linha-\nseguinte"
    assert desquebrar_safe(raw) == "linhaseguinte"
