from tradutor.quality_checks import format_quality_cell, run_translation_quality_checks


def test_quality_checks_flag_glossary_leak_and_missing_canonical() -> None:
    source = "The Four Holy Elders entered with the Sabre-Toothed Tigers."
    translated = "Os Quatro Anciãos Sagrados entraram com as Sabre-Toothed Tigers."
    terms = [
        {
            "key": "Four Holy Elders",
            "pt": "Quatro Santos",
            "aliases": ["Quatro Anciãos Sagrados"],
            "enforce": True,
        },
        {
            "key": "Sabre-Toothed Tigers",
            "pt": "Tigres Dente-de-Sabre",
            "aliases": ["Sabre-Toothed Tigers"],
            "enforce": True,
        },
    ]

    report = run_translation_quality_checks(source, translated, terms)

    issue_types = {issue["type"] for issue in report["issues"]}
    assert "source_term_in_target" in issue_types
    assert "missing_canonical_term" in issue_types
    assert report["score"] < 100


def test_quality_checks_flag_bad_name_alias() -> None:
    source = "Suou Kayako bowed her head."
    translated = "Kayado baixou a cabeça."
    terms = [
        {
            "key": "Suou Kayako",
            "pt": "Suou Kayako",
            "category": "personagem",
            "gender": "feminino",
            "bad_aliases": ["Kayado"],
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"]["bad_alias_in_target"] == 1


def test_quality_checks_flag_possible_gender_mismatch() -> None:
    source = "Abis Angun was taller than her brother."
    translated = "Abis Angun era mais alto que seu irmão."
    terms = [
        {
            "key": "Abis Angun",
            "pt": "Abis Angun",
            "category": "personagem",
            "gender": "feminino",
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"]["possible_gender_mismatch"] == 1


def test_quality_checks_flag_mixed_gender_adjectives_without_name() -> None:
    source = "Be careful about what comes next."
    translated = "É bom que você esteja sendo cuidadosa, atento ao que vem a seguir."

    report = run_translation_quality_checks(source, translated, [])

    assert report["issues_by_type"]["possible_gender_mismatch"] == 1


def test_quality_checks_does_not_flag_masculine_noun_near_feminine_name() -> None:
    source = "Suou Kayako looked embarrassed and guilty."
    translated = "Suou Kayako estava parada, com um olhar envergonhado e culpado."
    terms = [
        {
            "key": "Suou Kayako",
            "pt": "Suou Kayako",
            "category": "personagem",
            "gender": "feminino",
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"].get("possible_gender_mismatch") is None


def test_quality_checks_does_not_flag_alias_inside_canonical_translation() -> None:
    source = "Goddess Vicius smiled."
    translated = "A Deusa Vicius sorriu."
    terms = [
        {
            "key": "Goddess Vicius",
            "pt": "Deusa Vicius",
            "aliases": ["Vicius"],
            "enforce": True,
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"].get("source_term_in_target") is None


def test_quality_checks_allows_explicit_target_alias() -> None:
    source = "The Wildly Beautiful Emperor arrived."
    translated = "Zine chegou ao acampamento."
    terms = [
        {
            "key": "Wildly Beautiful Emperor",
            "pt": "Imperador Selvagemente Belo",
            "source_aliases": ["Beautiful Wild Emperor"],
            "allowed_target_aliases": ["Zine"],
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"].get("source_term_in_target") is None


def test_quality_checks_flags_bad_alias_separately_from_source_alias() -> None:
    source = "The Children of Vicius were summoned."
    translated = "Os Discípulos de Vicius foram convocados."
    terms = [
        {
            "key": "Children of Vicius",
            "pt": "Filhos de Vicius",
            "source_aliases": ["Vicius's Disciples"],
            "bad_aliases": ["Discípulos de Vicius"],
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"]["bad_alias_in_target"] == 1


def test_quality_checks_does_not_flag_fixed_expression_de_surpresa() -> None:
    source = "Kirihara was caught off guard."
    translated = "Mesmo tendo sido pego de surpresa, Kirihara permaneceu quieto."
    terms = [
        {
            "key": "Kirihara Takuto",
            "pt": "Kirihara Takuto",
            "aliases": ["Kirihara"],
            "category": "personagem",
            "gender": "masculino",
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"].get("possible_gender_mismatch") is None


def test_quality_checks_clean_text_scores_high() -> None:
    source = "The Dragonslayer spoke."
    translated = "O Matador de Dragões falou."
    terms = [{"key": "Dragonslayer", "pt": "Matador de Dragões", "enforce": True}]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["score"] == 100
    assert format_quality_cell(report) == "100/100"


def test_quality_checks_flag_refine_marker() -> None:
    report = run_translation_quality_checks(
        "Source",
        "### TEXTO_REFINADO_INICIO\nTexto.\n### TEXTO_REFINADO_FIM",
        [],
    )

    assert report["issues_by_type"]["residual_translation_marker"] == 1


def test_quality_checks_flag_common_english_interjection() -> None:
    report = run_translation_quality_checks("Phew, I am thirsty.", "Phew, fiquei com sede. Huh?", [])

    assert report["issues_by_type"]["residual_english"] == 2
