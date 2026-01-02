from tradutor.cleanup import cleanup_before_refine


def _cleanup(text: str) -> str:
    cleaned, _ = cleanup_before_refine(text)
    return cleaned


def test_dedupe_keeps_three_crack():
    text = "Crack!\nCrack!\nCrack!"
    cleaned = _cleanup(text)
    assert cleaned.count("Crack!") == 3


def test_dedupe_keeps_double_question_dash():
    text = "— ?\n— ?"
    cleaned = _cleanup(text)
    assert cleaned.count("— ?") == 2


def test_dedupe_keeps_double_ellipsis_dash():
    text = "— …\n— …"
    cleaned = _cleanup(text)
    assert cleaned.count("— …") == 2
