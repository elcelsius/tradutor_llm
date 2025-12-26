from tradutor.refine import has_suspicious_repetition


def test_repetition_detector_not_trigger_on_normal_text() -> None:
    text = "Ele disse algo para Ayaka. Ela respondeu calmamente. O narrador descreve a cena em detalhes naturais sem loops."
    assert has_suspicious_repetition(text) is False


def test_repetition_detector_triggers_on_loops() -> None:
    para = "Banewolf riu alto e ergueu a mão de Ayaka."
    text = f"{para}\n\n{para}\n\n{para}"
    assert has_suspicious_repetition(text) is True
