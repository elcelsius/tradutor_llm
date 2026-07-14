from tradutor.desquebrar_safe import desquebrar_safe, safe_reflow


def test_join_simple_continuation():
    """Processamento interno auxiliar."""
    raw = "Primeira parte\ncontinua aqui."
    assert safe_reflow(raw) == "Primeira parte continua aqui."


def test_join_hyphenation_without_space():
    """Processamento interno auxiliar."""
    raw = "palavra-\nquebrada no meio"
    assert safe_reflow(raw) == "palavraquebrada no meio"


def test_short_line_not_treated_as_title():
    """Processamento interno auxiliar."""
    raw = "Em uma linha\ncontinua sem titulo"
    assert safe_reflow(raw) == "Em uma linha continua sem titulo"


def test_preserve_dialogue_and_blank_lines():
    """Processamento interno auxiliar."""
    raw = '"Oi."\nela respondeu.\n\n"Nova fala"\nsegue aqui'
    expected = '"Oi."\nela respondeu.\n\n"Nova fala"\nsegue aqui'
    assert safe_reflow(raw) == expected


def test_preserve_em_dash_dialogue_start():
    """Processamento interno auxiliar."""
    raw = "Linha anterior\n— Então comecou.\ncontinua aqui"
    expected = "Linha anterior\n— Então comecou.\ncontinua aqui"
    assert safe_reflow(raw) == expected


def test_block_join_when_next_is_uppercase_or_title():
    """Processamento interno auxiliar."""
    raw = "final de frase\nProximo Paragrafo\nCAPITULO UM\ntexto inicia"
    expected = "final de frase\nProximo Paragrafo\nCAPITULO UM\ntexto inicia"
    assert safe_reflow(raw) == expected


def test_desquebrar_safe_wrapper():
    """Processamento interno auxiliar."""
    raw = "linha-\nseguinte"
    assert desquebrar_safe(raw) == "linhaseguinte"
