from tradutor.glossary_audit import (
    audit_glossary_data,
    format_audit_report,
    is_probably_portuguese_alias,
)


def test_glossary_audit_finds_ambiguous_and_portuguese_source_aliases() -> None:
    """Processamento interno auxiliar."""
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
    """Processamento interno auxiliar."""
    assert is_probably_portuguese_alias("Quatro Anciãos Sagrados")
    assert is_probably_portuguese_alias("Rei Matador de Monstros")
    assert not is_probably_portuguese_alias("Mya-a-ah")
    assert not is_probably_portuguese_alias("Lord of the Flies")


def test_glossary_audit_previews_redundant_source_aliases() -> None:
    report = audit_glossary_data(
        {
            "terms": [
                {
                    "key": "E-Class Hero",
                    "pt": "Herói de Classe E",
                    "source_aliases": ["E-class Hero"],
                }
            ]
        }
    )

    assert report["summary"]["redundant_source_aliases"] == 1
    assert "Redundant source aliases:" in format_audit_report(report)
