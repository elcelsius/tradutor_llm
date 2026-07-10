from tradutor.language_guardrails import detect_residual_english, english_leak_segments


def test_detect_residual_english_long_sentence() -> None:
    text = "“I have no desire to die,” replied Seras calmly, choosing not to answer directly."

    detected, reason = detect_residual_english(text)

    assert detected
    assert reason.startswith("residual_english:")


def test_does_not_flag_natural_portuguese_with_names() -> None:
    text = "Sogou Ayaka olhou para Seras com calma e respondeu que não pretendia recuar."

    assert english_leak_segments(text) == []
