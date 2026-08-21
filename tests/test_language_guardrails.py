from tradutor.language_guardrails import detect_residual_english, english_leak_segments


def test_detect_residual_english_long_sentence() -> None:
    """Processamento interno auxiliar."""
    text = "“I have no desire to die,” replied Seras calmly, choosing not to answer directly."

    detected, reason = detect_residual_english(text)

    assert detected
    assert reason.startswith("residual_english:")


def test_detect_residual_english_known_single_game_term() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english("Ela perdeu os buffs e recuou.")

    assert detected
    assert reason == "residual_english:buffs"


def test_detect_residual_english_plural_self() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english(
        "As máscaras eram selves temporários para um propósito."
    )

    assert detected
    assert reason == "residual_english:selves"


def test_detect_residual_english_slang_abbreviation() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english("Ela é super desconfiada AF.")

    assert detected
    assert reason == "residual_english:AF"


def test_detect_residual_english_single_word_connector() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english("Ela é arrogante, though.")

    assert detected
    assert reason == "residual_english:though"


def test_detect_residual_english_embedded_short_connector() -> None:
    detected, reason = detect_residual_english("Ela recusaria, but você se saiu bem.")

    assert detected
    assert reason == "residual_english:but"


def test_detect_residual_english_stylized_alright() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english("“Arright!”, disse Itsuki.")

    assert detected
    assert reason == "residual_english:Arright"


def test_detect_residual_english_hybrid_pronoun() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english(
        "Ninguém o encarava com raiva — they todos pareciam felizes."
    )

    assert detected
    assert reason == "residual_english:they"


def test_detect_residual_english_boost() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english("Eles precisavam de um boost extra.")

    assert detected
    assert reason == "residual_english:boost"


def test_detect_residual_english_short_phrase() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english("“I see”, disse Hijiri.")

    assert detected
    assert reason == "residual_english:I see"


def test_detect_residual_english_hybrid_short_phrase() -> None:
    """Detecta uma resposta curta parcialmente traduzida."""
    detected, reason = detect_residual_english("“Eu see”, disse Hijiri.")

    assert detected
    assert reason == "residual_english:Eu see"


def test_detect_residual_english_hybrid_pronoun_with_english_verb() -> None:
    detected, reason = detect_residual_english("Eu have no desire to die.")

    assert detected
    assert reason.startswith("residual_english:Eu have")


def test_detect_residual_english_embedded_in_portuguese_sentence() -> None:
    """Detecta uma frase inglesa que ficou inserida em uma fala em PT-BR."""
    text = (
        "No fundo, essa pessoa não se importa com Vicius — "
        "but the Goddess trusts them all the same."
    )

    detected, reason = detect_residual_english(text)

    assert detected
    assert reason.startswith("residual_english:but the Goddess")


def test_embedded_english_takes_priority_over_single_token_leak() -> None:
    """O reparo deve receber a frase inteira, não só o token ``they``."""
    text = (
        "A unidade da Asagi—they will be under your command, mas agirão "
        "independentemente de nós."
    )

    segments = english_leak_segments(text)

    assert segments
    assert segments[0] == "they will be under your command"


def test_detect_residual_english_mixed_pronoun_artifact() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english("I não me importo de voltar para casa.")

    assert detected
    assert reason == "residual_english:I"


def test_detect_residual_english_mixed_pronoun_before_any_portuguese_word() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english("Uau—I aposto que ela consegue.")

    assert detected
    assert reason == "residual_english:I"


def test_does_not_flag_portuguese_stutter_that_starts_with_i() -> None:
    detected, reason = detect_residual_english("“I-impossível… É v-você…”")

    assert detected is False
    assert reason == ""


def test_detects_stuttered_english_word_after_i() -> None:
    detected, reason = detect_residual_english("“I-impossible… It cannot be.”")

    assert detected
    assert reason == "residual_english:I-"


def test_detects_repeated_english_i_stutter() -> None:
    detected, reason = detect_residual_english("“I-I will not surrender.”")

    assert detected
    assert reason == "residual_english:I-"


def test_detect_residual_english_interjection() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english("Uhh… sei disso.")

    assert detected
    assert reason == "residual_english:Uhh"


def test_detect_residual_english_short_interjection() -> None:
    """Processamento interno auxiliar."""
    detected, reason = detect_residual_english("Uh… sei disso.")

    assert detected
    assert reason == "residual_english:Uh"


def test_does_not_flag_natural_portuguese_with_names() -> None:
    """Processamento interno auxiliar."""
    text = (
        "Sogou Ayaka olhou para Seras com calma e respondeu que não pretendia recuar."
    )

    assert english_leak_segments(text) == []
