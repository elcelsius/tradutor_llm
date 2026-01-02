from tradutor.translate import enforce_canonical_terms


def test_enforce_canonical_glossary_term():
    terms = [
        {
            "key": "Lord of the Flies",
            "pt": "Senhor das Moscas",
            "aliases": ["Lord-of-the-Flies"],
            "enforce": True,
        }
    ]
    text = "O Lord of the Flies apareceu. Outro Lord-of-the-Flies caiu."

    normalized, replacements = enforce_canonical_terms(text, terms)

    assert "Senhor das Moscas" in normalized
    assert "Lord of the Flies" not in normalized
    assert replacements.get("Lord of the Flies", 0) >= 1
