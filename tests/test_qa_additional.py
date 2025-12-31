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
