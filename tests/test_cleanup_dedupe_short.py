from tradutor.cleanup import cleanup_before_refine


def _cleanup(text: str) -> str:
    """Processamento interno auxiliar."""
    cleaned, _ = cleanup_before_refine(text)
    return cleaned


def test_dedupe_keeps_three_crack():
    """Processamento interno auxiliar."""
    text = "Crack!\nCrack!\nCrack!"
    cleaned = _cleanup(text)
    assert cleaned.count("Crack!") == 3


def test_dedupe_keeps_double_question_dash():
    """Processamento interno auxiliar."""
    text = "— ?\n— ?"
    cleaned = _cleanup(text)
    assert cleaned.count("— ?") == 2


def test_dedupe_keeps_double_ellipsis_dash():
    """Processamento interno auxiliar."""
    text = "— …\n— …"
    cleaned = _cleanup(text)
    assert cleaned.count("— …") == 2


def test_dedupe_keeps_short_fragments_in_paragraph():
    """Processamento interno auxiliar."""
    text = "Crack! Crack! Crack!"
    cleaned = _cleanup(text)
    assert cleaned.count("Crack!") == 3


def test_dedupe_keeps_short_dialogue_fragments():
    """Processamento interno auxiliar."""
    text = "— ? — ?"
    cleaned = _cleanup(text)
    assert cleaned.count("— ?") == 2
