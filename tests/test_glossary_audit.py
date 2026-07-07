from tradutor.glossary_audit import audit_glossary_data, is_probably_portuguese_alias


def test_glossary_audit_finds_ambiguous_and_portuguese_source_aliases() -> None:
    data = {
        "terms": [
            {
                "key": "Belzegea",
                "pt": "Belzegea",
                "aliases": ["Lord of the Flies", "Senhor das Moscas"],
            },
            {
                "key": "Lord of the Flies",
                "pt": "Senhor das Moscas",
                "aliases": ["Lord-of-the-Flies"],
            },
        ]
    }

    report = audit_glossary_data(data)

    assert report["summary"]["terms"] == 2
    assert report["summary"]["ambiguous_source_aliases"] == 1
    assert report["ambiguous_source_aliases"][0]["value"] == "Lord of the Flies"
    assert report["summary"]["portuguese_source_aliases"] == 1
    assert report["portuguese_source_aliases"][0]["alias"] == "Senhor das Moscas"


def test_portuguese_alias_heuristic_ignores_english_noise() -> None:
    assert is_probably_portuguese_alias("Quatro Anciãos Sagrados")
    assert is_probably_portuguese_alias("Rei Matador de Monstros")
    assert not is_probably_portuguese_alias("Mya-a-ah")
    assert not is_probably_portuguese_alias("Lord of the Flies")
