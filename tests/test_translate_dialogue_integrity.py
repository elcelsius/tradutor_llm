from tradutor.qa import needs_retry
from tradutor.postprocess_translation import postprocess_translation


def test_needs_retry_unbalanced_curly_quotes():
    input_text = "“How did you get there?”"
    output_text = "“Como você chegou lá?"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is True
    assert "unbalanced_quotes" in reason


def test_needs_retry_extra_short_repetition():
    input_text = "Crack!\nSilence."
    output_text = "Crack!\nCrack!\nCrack!\nSilence."

    retry, reason = needs_retry(input_text, output_text)

    assert retry is True
    assert "extra_short_repetition" in reason


def test_dash_line_strips_trailing_quote():
    samples = [
        ("— Entendido”, Seras me respondeu.", "— Entendido, Seras me respondeu."),
        ("— Oh?!” Vicius perguntou.", "— Oh?! Vicius perguntou."),
        ("— Eh?” Vicius disse.", "— Eh? Vicius disse."),
        ("— Hm.” Seras comentou.", "— Hm. Seras comentou."),
        ("— …” ele murmurou.", "— … ele murmurou."),
    ]

    for raw, expected in samples:
        cleaned = postprocess_translation(raw, en_text="")  # en_text vazio para passar pelo pipeline
        assert cleaned == expected
