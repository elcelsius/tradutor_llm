from tradutor.text_postprocess import apply_custom_normalizers


def test_custom_normalizer_canonizes_touka_variants() -> None:
    """Processamento interno auxiliar."""
    text = "Too-ka encontrou Touka e Tou-ka."
    out = apply_custom_normalizers(text)
    assert out.count("Touka") == 3
    assert "Too-ka" not in out
    assert "Tou-ka" not in out


def test_custom_normalizer_converts_gulp() -> None:
    """Processamento interno auxiliar."""
    text = "Gulp.\nThe baron swallowed hard and kept going."
    out = apply_custom_normalizers(text)
    assert out.startswith("Glup.")
    assert "swallowed hard" in out


def test_custom_normalizer_translates_common_english_interjections() -> None:
    """Processamento interno auxiliar."""
    text = "Phew, me deu sede.\n\nGeez, isso foi estranho.\n\nHuh, entendi.\n\nUgh, que saco."
    out = apply_custom_normalizers(text)
    assert "Ufa, me deu sede." in out
    assert "Nossa, isso foi estranho." in out
    assert "Hã, entendi." in out
    assert "Argh, que saco." in out
    assert "Phew" not in out
    assert "Geez" not in out
    assert "Huh" not in out
    assert "Ugh" not in out


def test_custom_normalizer_translates_whoa_and_generic_skill() -> None:
    text = "Whoa! Essa é a melhor skill; essas skills vão ajudar."

    out = apply_custom_normalizers(text, convert_quote_dialogues=False)

    assert out == "Uau! Essa é a melhor habilidade; essas habilidades vão ajudar."


def test_custom_normalizer_translates_embedded_english_connectors() -> None:
    text = "And assim seguimos, or pelo menos tentamos, but você sabe disso."

    out = apply_custom_normalizers(text, convert_quote_dialogues=False)

    assert out == "E assim seguimos, ou pelo menos tentamos, mas você sabe disso."


def test_custom_normalizer_quotes_to_dash() -> None:
    """Processamento interno auxiliar."""
    text = '“Hello there.”\nNarration line.\n"Oi!"'
    out = apply_custom_normalizers(text)
    lines = out.splitlines()
    assert lines[0].startswith("— Hello")
    assert lines[1] == "Narration line."
    assert lines[2].startswith("— Oi")


def test_custom_normalizer_can_preserve_quote_dialogues() -> None:
    """Processamento interno auxiliar."""
    text = '“Hello there.”\nNarration line.\n"Oi!"'
    out = apply_custom_normalizers(text, convert_quote_dialogues=False)
    lines = out.splitlines()
    assert lines[0] == "“Hello there.”"
    assert lines[1] == "Narration line."
    assert lines[2] == '"Oi!"'


def test_custom_normalizer_fixes_poderam() -> None:
    """Processamento interno auxiliar."""
    text = "Eles poderam vencer."
    out = apply_custom_normalizers(text)
    assert "puderam" in out


def test_custom_normalizer_merges_speech_with_verb() -> None:
    """Processamento interno auxiliar."""
    text = "“Oi.”\n\nperguntou Joao."
    out = apply_custom_normalizers(text)
    assert "— Oi. perguntou Joao." in out


def test_custom_normalizer_merges_dash_attribution_line() -> None:
    """Processamento interno auxiliar."""
    text = "— Por quê?\n— perguntou o Kirihara, direto."
    out = apply_custom_normalizers(text, convert_quote_dialogues=False)
    assert out == "— Por quê? — perguntou o Kirihara, direto."


def test_custom_normalizer_merges_dash_attribution_after_blank_line() -> None:
    """Processamento interno auxiliar."""
    text = "— O que você está lendo, Sogou?\n\n— Oyamada provocou enquanto arrancava o livro."
    out = apply_custom_normalizers(text, convert_quote_dialogues=False)
    assert (
        out
        == "— O que você está lendo, Sogou? — Oyamada provocou enquanto arrancava o livro."
    )


def test_custom_normalizer_does_not_merge_pronoun_speech_as_attribution() -> None:
    """Processamento interno auxiliar."""
    text = "— O amor mais incrível que você vai ler este ano?!\n\n— Eu disse, devolva!"
    out = apply_custom_normalizers(text, convert_quote_dialogues=False)
    assert out == text


def test_custom_normalizer_merges_pareceu_attribution() -> None:
    """Processamento interno auxiliar."""
    text = "— H-heróis? E-eu também sou um herói?\n— Oyamada pareceu chocado."
    out = apply_custom_normalizers(text, convert_quote_dialogues=False)
    assert out == "— H-heróis? E-eu também sou um herói? — Oyamada pareceu chocado."
