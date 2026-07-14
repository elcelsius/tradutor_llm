from tradutor.qa import needs_retry


def test_needs_retry_detects_lowercased_fragment() -> None:
    ok, reason = needs_retry("Context", "Primeira linha.\n\no rosto dela.")
    assert ok is True
    assert reason == "lowercased_fragment"


def test_needs_retry_detects_lowercase_narration_start() -> None:
    ok, reason = needs_retry("Context", "ela olhou para a porta.")
    assert ok is True
    assert reason == "lowercase_narration_start"


def test_needs_retry_detects_truncated_ellipsis_token() -> None:
    ok, reason = needs_retry("Contexto", "Voc...")
    assert ok is True
    assert reason == "truncated_token_ellipsis"


def test_needs_retry_detects_merged_or_missing_paragraph() -> None:
    source = "“First line.”\n\nNarration between the lines.\n\n“Last line.”"
    translated = "“Primeira linha.” Narração entre as falas.\n\n“Última linha.”"

    retry, reason = needs_retry(source, translated)

    assert retry is True
    assert reason == "omissao_paragrafos (2/3)"


def test_needs_retry_allows_paragraphs_preserved_as_nonempty_lines() -> None:
    source = "First.\n\nSecond.\n\nThird."
    translated = "Primeiro.\nSegundo.\nTerceiro."

    retry, reason = needs_retry(source, translated)

    assert retry is False
    assert reason == ""
