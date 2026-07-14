from tradutor.quality_checks import format_quality_cell, run_translation_quality_checks


def test_quality_checks_flag_glossary_leak_and_missing_canonical() -> None:
    """Processamento interno auxiliar."""
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
    """Processamento interno auxiliar."""
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


def test_quality_checks_handles_case_only_bad_alias_without_flagging_canonical() -> (
    None
):
    """Processamento interno auxiliar."""
    terms = [{"key": "Kyokugen", "pt": "Kyokugen", "bad_aliases": ["kyokugen"]}]

    canonical = run_translation_quality_checks(
        "Kyokugen", "A técnica Kyokugen funciona.", terms
    )
    lowercase = run_translation_quality_checks(
        "Kyokugen", "A técnica kyokugen funciona.", terms
    )

    assert canonical["issues_by_type"].get("bad_alias_in_target") is None
    assert lowercase["issues_by_type"]["bad_alias_in_target"] == 1


def test_quality_checks_flag_possible_gender_mismatch() -> None:
    """Processamento interno auxiliar."""
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


def test_quality_checks_does_not_bind_object_name_to_speaker_adjective() -> None:
    """Processamento interno auxiliar."""
    terms = [
        {
            "key": "Kirihara Takuto",
            "pt": "Kirihara Takuto",
            "category": "personagem",
            "gender": "masculino",
        }
    ]

    report = run_translation_quality_checks(
        "Seras could not have defeated Kirihara alone.",
        "Eu não teria derrotado o Kirihara Takuto sozinha.",
        terms,
    )

    assert report["issues_by_type"].get("possible_gender_mismatch") is None


def test_quality_checks_flag_mixed_gender_adjectives_without_name() -> None:
    """Processamento interno auxiliar."""
    source = "Be careful about what comes next."
    translated = "É bom que você esteja sendo cuidadosa, atento ao que vem a seguir."

    report = run_translation_quality_checks(source, translated, [])

    assert report["issues_by_type"]["possible_gender_mismatch"] == 1


def test_quality_checks_does_not_flag_masculine_noun_near_feminine_name() -> None:
    """Processamento interno auxiliar."""
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
    """Processamento interno auxiliar."""
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
    """Processamento interno auxiliar."""
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
    assert report["issues_by_type"].get("missing_canonical_term") is None


def test_quality_checks_allows_contextual_noun_form_for_skill() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks(
        "Touka used Paralyze.",
        "O alcance é igual ao da Paralisia.",
        [
            {
                "key": "Paralyze",
                "pt": "Paralisar",
                "source_case_sensitive": True,
                "allowed_target_aliases": ["Paralisia"],
            }
        ],
    )

    assert report["issues_by_type"].get("missing_canonical_term") is None


def test_quality_checks_flags_bad_alias_separately_from_source_alias() -> None:
    """Processamento interno auxiliar."""
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


def test_quality_checks_flags_contextual_target_replacement_alias() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks(
        "After the Deathmatch",
        "Após a Batalha de Morte.",
        [
            {
                "key": "Deathmatch",
                "pt": "Combate Mortal",
                "target_replacements": {
                    "Após a Batalha de Morte": "Após o Combate Mortal"
                },
            }
        ],
    )

    assert report["issues_by_type"]["bad_alias_in_target"] == 1


def test_quality_checks_respects_case_sensitive_source_term() -> None:
    """Processamento interno auxiliar."""
    terms = [{"key": "Freeze", "pt": "Congelar", "source_case_sensitive": True}]

    common_word_report = run_translation_quality_checks(
        "After she watched his body freeze, she lost consciousness.",
        "Depois que ela viu o corpo dele ficar imóvel, perdeu a consciência.",
        terms,
    )
    skill_report = run_translation_quality_checks(
        "Touka cast Freeze on the undead enemy.",
        "Touka lançou a habilidade no inimigo morto-vivo.",
        terms,
    )

    assert common_word_report["issues_by_type"].get("missing_canonical_term") is None
    assert skill_report["issues_by_type"]["missing_canonical_term"] == 1


def test_quality_checks_does_not_require_full_canonical_form_for_source_alias() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks(
        "The refugees are headed to Yonato.",
        "Os refugiados estão indo para Yonato.",
        [
            {
                "key": "State of Yonato",
                "pt": "Estado de Yonato",
                "source_aliases": ["Yonato"],
            }
        ],
    )

    assert report["issues_by_type"].get("missing_canonical_term") is None


def test_quality_checks_requires_canonical_form_for_enforced_source_alias() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks(
        "The Kurosaga Clan arrived.",
        "O grupo da Munin chegou.",
        [
            {
                "key": "Forbidden Words Clan",
                "pt": "Clã das Palavras Proibidas",
                "source_aliases": ["Kurosaga Clan"],
                "enforce": True,
            }
        ],
    )

    assert report["issues_by_type"]["missing_canonical_term"] == 1


def test_quality_checks_does_not_flag_fixed_expression_de_surpresa() -> None:
    """Processamento interno auxiliar."""
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
    """Processamento interno auxiliar."""
    source = "The Dragonslayer spoke."
    translated = "O Matador de Dragões falou."
    terms = [{"key": "Dragonslayer", "pt": "Matador de Dragões", "enforce": True}]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["score"] == 100
    assert format_quality_cell(report) == "100/100"


def test_quality_checks_flag_refine_marker() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks(
        "Source",
        "### TEXTO_REFINADO_INICIO\nTexto.\n### TEXTO_REFINADO_FIM",
        [],
    )

    assert report["issues_by_type"]["residual_translation_marker"] == 1


def test_quality_checks_flag_common_english_interjection() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks(
        "Phew, I am thirsty.", "Phew, fiquei com sede. Huh?", []
    )

    assert report["issues_by_type"]["residual_english"] == 2


def test_quality_checks_flags_known_game_jargon_left_in_english() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks(
        "The buffs faded.", "Os buffs desapareceram.", []
    )

    assert report["issues_by_type"]["residual_english"] == 1


def test_quality_checks_flag_malformed_quote_boundary() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks("Source", "”“Ah, tudo bem.”", [])

    assert report["issues_by_type"]["malformed_quote_boundary"] == 1


def test_quality_checks_allows_dialogues_separated_by_newlines() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks("Source", "“Oi.”\n\n“Tudo bem.”", [])

    assert report["issues_by_type"].get("malformed_quote_boundary") is None


def test_quality_checks_flags_missing_quote_spacing() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks("Source", '"Uma fala.""Outra fala."', [])

    assert report["issues_by_type"]["missing_quote_spacing"] == 1


def test_quality_checks_flags_stray_format_marker_after_dialogue() -> None:
    """Processamento interno auxiliar."""
    report = run_translation_quality_checks("Source", "“Tudo bem.”*", [])

    assert report["issues_by_type"]["stray_format_marker"] == 1
