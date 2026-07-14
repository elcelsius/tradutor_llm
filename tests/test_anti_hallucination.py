from tradutor.anti_hallucination import (
    detect_entity_mutation,
    detect_inline_slash_mutation,
    detect_language_anomaly,
    detect_repetition_anomaly,
)


def test_repetition_guard_allows_normal_portuguese_function_words() -> None:
    """Processamento interno auxiliar."""
    text = " ".join(
        [
            "A personagem olhou para a porta e disse que não queria entrar na sala.",
            "Depois, ela explicou que a decisão era importante para todos os presentes.",
        ]
        * 8
    )

    assert not detect_repetition_anomaly(text)


def test_repetition_guard_flags_dominant_relevant_token() -> None:
    """Processamento interno auxiliar."""
    text = " ".join(["cristal"] * 20 + ["personagem", "caminhou", "lentamente"] * 10)

    assert detect_repetition_anomaly(text)


def test_language_guard_allows_portuguese_cognates_and_flags_french() -> None:
    """Processamento interno auxiliar."""
    assert not detect_language_anomaly("Esta decisão existe porque a situação é grave.")
    assert detect_language_anomaly("Bonjour, mon ami.")


def test_entity_and_inline_slash_guards_detect_model_mutations() -> None:
    """Processamento interno auxiliar."""
    original = "Sogou falou com Sogou, Munin e Munin. Hijiri observou Hijiri."
    candidate = "Sogamente falou com Munamente. Hijenci escreveu do/a mundo."

    assert detect_entity_mutation(original, candidate)
    assert detect_inline_slash_mutation(original, candidate)
