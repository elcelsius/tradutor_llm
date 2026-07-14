from tradutor.qa import needs_retry


def test_needs_retry_detects_ellipsis_in_word() -> None:
    """Processamento interno auxiliar."""
    ok, reason = needs_retry(
        "The original sentence is complete.", "A frase incom...pleta em PT."
    )
    assert ok is True
    assert reason == "ellipsis_in_word"


def test_needs_retry_detects_excessive_ellipsis() -> None:
    """Processamento interno auxiliar."""
    ok, reason = needs_retry("Texto sem reticencias.", "Texto ... com ... omissoes.")
    assert ok is True
    assert reason in {"ellipsis_in_word", "ellipsis_suspect"}


def test_needs_retry_allows_literary_ellipsis_in_dialogue() -> None:
    """Processamento interno auxiliar."""
    ok, reason = needs_retry(
        "“I am sorry...” Kayako said. A moment before, she thought it was impossible, but...",
        "“Me desculpe…” disse Kayako. Alguns momentos antes, ela teria achado impossível, mas… deveria?",
    )
    assert ok is False
    assert reason == ""
