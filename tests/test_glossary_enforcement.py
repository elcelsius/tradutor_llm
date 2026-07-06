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


def test_enforce_bad_alias_without_expanding_valid_aliases():
    terms = [
        {
            "key": "Banewolf",
            "pt": "Banewolf",
            "aliases": ["Bane"],
            "bad_aliases": ["Banamente"],
        }
    ]
    text = "Banamente riu. Bane-san pegou a garrafa."

    normalized, replacements = enforce_canonical_terms(text, terms)

    assert "Banewolf riu." in normalized
    assert "Bane-san" in normalized
    assert "Banamente" not in normalized
    assert replacements == {"Banamente": 1}
