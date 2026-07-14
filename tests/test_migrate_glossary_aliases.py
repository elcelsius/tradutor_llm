from scripts.migrate_glossary_aliases import migrate


def _term(data: dict, key: str) -> dict:
    """Processamento interno auxiliar."""
    for entry in data["terms"]:
        if entry["key"] == key:
            return entry
    raise AssertionError(f"term not found: {key}")


def test_migrate_separates_ambiguous_belzegea_aliases() -> None:
    """Processamento interno auxiliar."""
    data = {
        "terms": [
            {
                "key": "Belzegea",
                "pt": "Belzegea",
                "aliases": ["Lord of the Flies", "Fly Guy", "Senhor das Moscas"],
            }
        ]
    }

    summary = migrate(data)
    term = _term(data, "Belzegea")

    assert summary["terms_updated"] >= 1
    assert term["source_aliases"] == ["Fly Guy"]
    assert term["aliases"] == ["Fly Guy"]
    assert "Senhor das Moscas" in term["allowed_target_aliases"]
    assert "Lord of the Flies" not in term["source_aliases"]


def test_migrate_moves_clear_noncanonical_target_forms_to_bad_aliases() -> None:
    """Processamento interno auxiliar."""
    data = {
        "terms": [
            {
                "key": "Monster Slayer King",
                "pt": "Rei Caçador de Monstros",
                "aliases": [
                    "King of Monster Slayers",
                    "Rei Matador de Monstros",
                    "Rei Exterminador de Monstros",
                ],
            }
        ]
    }

    migrate(data)
    term = _term(data, "Monster Slayer King")

    assert term["source_aliases"] == ["King of Monster Slayers"]
    assert "Rei Matador de Monstros" in term["bad_aliases"]
    assert "Rei Exterminador de Monstros" in term["bad_aliases"]
    assert "Rei Matador de Monstros" not in term.get("allowed_target_aliases", [])


def test_migrate_removes_cross_term_character_aliases() -> None:
    """Processamento interno auxiliar."""
    data = {
        "terms": [
            {
                "key": "Anael",
                "pt": "Anael",
                "aliases": ["Erika Anaorbael", "Mistress Anael"],
            },
            {
                "key": "Erika Anaorbael",
                "pt": "Erika Anaorbael",
                "aliases": ["Forbidden Witch", "Erika"],
            },
            {
                "key": "Forbidden Witch",
                "pt": "Bruxa Proibida",
                "aliases": ["the Forbidden Witch"],
            },
            {"key": "Lil'ella", "pt": "Lil'ella", "aliases": []},
            {"key": "Lokiella", "pt": "Lokiella", "aliases": ["Lil'ella", "Loki-ella"]},
        ]
    }

    migrate(data)

    assert _term(data, "Anael")["source_aliases"] == ["Mistress Anael"]
    assert _term(data, "Erika Anaorbael")["source_aliases"] == ["Erika"]
    assert _term(data, "Lokiella")["source_aliases"] == ["Loki-ella"]


def test_migrate_removes_redundant_source_alias_equal_to_key() -> None:
    """Processamento interno auxiliar."""
    data = {
        "terms": [
            {
                "key": "Lord of the Flies",
                "pt": "Senhor das Moscas",
                "aliases": ["Lord of the Flies", "Lord-of-the-Flies"],
            }
        ]
    }

    migrate(data)

    assert _term(data, "Lord of the Flies")["source_aliases"] == ["Lord-of-the-Flies"]
