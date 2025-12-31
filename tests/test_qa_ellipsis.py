from tradutor.qa import needs_retry


def test_needs_retry_detects_ellipsis_in_word() -> None:
    ok, reason = needs_retry("The original sentence is complete.", "A frase incom...pleta em PT.")
    assert ok is True
    assert reason == "ellipsis_in_word"


def test_needs_retry_detects_excessive_ellipsis() -> None:
    ok, reason = needs_retry("Texto sem reticencias.", "Texto ... com ... omissoes.")
    assert ok is True
    assert reason in {"ellipsis_in_word", "ellipsis_suspect"}
