from tradutor.text_postprocess import apply_custom_normalizers


def test_custom_normalizer_canonizes_touka_variants() -> None:
    text = "Too-ka encontrou Touka e Tou-ka."
    out = apply_custom_normalizers(text)
    assert out.count("Touka") == 3
    assert "Too-ka" not in out
    assert "Tou-ka" not in out


def test_custom_normalizer_converts_gulp() -> None:
    text = "Gulp.\nThe baron swallowed hard and kept going."
    out = apply_custom_normalizers(text)
    assert out.startswith("Glup.")
    assert "swallowed hard" in out


def test_custom_normalizer_quotes_to_dash() -> None:
    text = "“Hello there.”\nNarration line.\n\"Oi!\""
    out = apply_custom_normalizers(text)
    lines = out.splitlines()
    assert lines[0].startswith("— Hello")
    assert lines[1] == "Narration line."
    assert lines[2].startswith("— Oi")


def test_custom_normalizer_fixes_poderam() -> None:
    text = "Eles poderam vencer."
    out = apply_custom_normalizers(text)
    assert "puderam" in out


def test_custom_normalizer_merges_speech_with_verb() -> None:
    text = "“Oi.”\n\nperguntou Joao."
    out = apply_custom_normalizers(text)
    assert "— Oi. perguntou Joao." in out
