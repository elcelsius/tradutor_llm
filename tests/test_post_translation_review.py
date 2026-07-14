from tradutor.post_translation_review import finalize_translation_text, review_translation_text
from tradutor.translate import ensure_section_heading, source_heading_to_pt


def test_source_heading_to_pt() -> None:
    assert source_heading_to_pt("Chapter 5:") == "# Capítulo 5:"
    assert source_heading_to_pt("Epilogue") == "# Epílogo"
    assert source_heading_to_pt("Afterword") == "# Pós-escrito"
    assert source_heading_to_pt("Full Text") is None


def test_ensure_section_heading_inserts_missing_heading() -> None:
    text, changed = ensure_section_heading("Minha consciência voltou à tona.", "Chapter 5:")

    assert changed is True
    assert text.startswith("# Capítulo 5:\n\nMinha consciência")


def test_ensure_section_heading_does_not_duplicate_heading() -> None:
    text, changed = ensure_section_heading("# Capítulo 2:\n\nTexto.", "Chapter 2:")

    assert changed is False
    assert text == "# Capítulo 2:\n\nTexto."


def test_review_does_not_restore_generic_heading_when_titled_heading_exists() -> None:
    text = "# Capítulo 1: Após o Combate Mortal\n\nDepois que Sogou acordou."

    reviewed, report = review_translation_text(text, sections=[{"title": "Chapter 1:"}])

    assert reviewed.count("# Capítulo 1:") == 1
    assert report.heading_fixes == 0


def test_review_translation_restores_headings_and_applies_alias_fixes() -> None:
    text = (
        "Prólogo em andamento.\n\n"
        "Após o confronto letal, Sogou dormiu.\n\n"
        "Nyaki está aqui, meow! Meow de novo.\n\n"
        "A Deusa-chin é sus AF. Eu wish que ela parasse—or fugisse. Era isso—, talvez.\n\n"
        "O Asagi acha isso estranho. Não confio no Asagi como aliado.\n\n"
        "Os Discípulos de Vicius chegaram.\n\n"
        "O monstro bipede sonha com semi-deuses.\n\n"
        "Não parece que a Deusa Vicius está manipulando Kashima Kobato, nem que ela está sendo arrastada.\n\n"
        "“Loki… Ella… Th-the… y’re…”\n\n"
        "Minha consciência voltou à tona."
    )
    sections = [
        {"title": "Chapter 1:"},
        {"title": "Chapter 5:"},
    ]
    terms = [
        {
            "key": "Children of Vicius",
            "pt": "Filhos de Vicius",
            "bad_aliases": ["Discípulos de Vicius"],
        },
        {
            "key": "Ikusaba Asagi",
            "pt": "Ikusaba Asagi",
            "category": "personagem",
            "gender": "feminino",
            "source_aliases": ["Asagi"],
        },
        {
            "key": "Mimori Touka",
            "pt": "Mimori Touka",
            "category": "personagem",
            "gender": "masculino",
            "source_aliases": ["Mimori"],
        }
    ]

    reviewed, report = review_translation_text(text, sections=sections, glossary_terms=terms)

    assert "# Capítulo 1:" in reviewed
    assert "# Capítulo 5:" in reviewed
    assert "miau" in reviewed
    assert "Miau de novo" in reviewed
    assert "Deusazinha" in reviewed
    assert "suspeita pra caramba" in reviewed
    assert "Quem me dera" in reviewed
    assert "parasse — ou fugisse" in reviewed
    assert "isso —, talvez" in reviewed
    assert "bípede" in reviewed
    assert "semideuses" in reviewed
    assert "El-eles…" in reviewed
    assert "A Asagi acha" in reviewed
    assert "na Asagi como aliada" in reviewed
    reviewed_male, _ = review_translation_text("A Mimori como aliada.", glossary_terms=terms)
    assert reviewed_male == "O Mimori como aliado."
    assert "manipulando a Asagi" in reviewed
    assert "Filhos de Vicius" in reviewed
    assert report.heading_fixes == 2
    assert report.glossary_replacements["Discípulos de Vicius->Filhos de Vicius"] == 1


def test_review_fixes_gendered_articles_for_feminine_creatures() -> None:
    terms = [{"key": "Slei", "pt": "Slei", "category": "criatura", "gender": "feminino"}]

    reviewed, _ = review_translation_text("O Slei avançou nas costas do Slei.", glossary_terms=terms)

    assert reviewed == "A Slei avançou nas costas da Slei."


def test_review_applies_contextual_glossary_target_replacement() -> None:
    terms = [
        {
            "key": "Deathmatch",
            "pt": "Combate Mortal",
            "target_replacements": {
                "Após a Batalha de Morte": "Após o Combate Mortal",
            },
            "bad_aliases": ["Deathmatch"],
        }
    ]

    reviewed, report = review_translation_text(
        "# Capítulo 1: Após a Batalha de Morte\n\nDepois do Deathmatch.",
        glossary_terms=terms,
    )

    assert "Após o Combate Mortal" in reviewed
    assert "Depois do Combate Mortal" in reviewed
    assert report.glossary_replacements["Após a Batalha de Morte->Após o Combate Mortal"] == 1


def test_review_collapses_duplicate_canonical_character_names() -> None:
    text = (
        "Sogou Sogou Ayaka falou com Takao Takao Hijiri.\n\n"
        "Ikusaba Asagi Asagi respondeu. Sogou Ayaka Sogou Ayaka voltou."
    )
    terms = [
        {
            "key": "Sogou Ayaka",
            "pt": "Sogou Ayaka",
            "category": "personagem",
            "source_aliases": ["Sogou", "Ayaka"],
        },
        {
            "key": "Takao Hijiri",
            "pt": "Takao Hijiri",
            "category": "personagem",
            "source_aliases": ["Hijiri"],
        },
        {
            "key": "Ikusaba Asagi",
            "pt": "Ikusaba Asagi",
            "category": "personagem",
            "source_aliases": ["Asagi"],
        },
    ]

    reviewed, report = review_translation_text(text, glossary_terms=terms)

    assert "Sogou Sogou Ayaka" not in reviewed
    assert "Takao Takao Hijiri" not in reviewed
    assert "Ikusaba Asagi Asagi" not in reviewed
    assert "Sogou Ayaka Sogou Ayaka" not in reviewed
    assert "Sogou Ayaka falou com Takao Hijiri" in reviewed
    assert "Ikusaba Asagi respondeu. Sogou Ayaka voltou." in reviewed
    assert report.text_replacements


def test_finalize_review_normalizes_all_caps_character_names_and_structure() -> None:
    text = "Após o Deathmatch, depois que SOGOU viu SERAS, ela se acalmou."
    terms = [
        {
            "key": "Sogou Ayaka",
            "pt": "Sogou Ayaka",
            "category": "personagem",
            "source_aliases": ["Sogou"],
        },
        {
            "key": "Seras Ashrain",
            "pt": "Seras Ashrain",
            "category": "personagem",
            "source_aliases": ["Seras"],
        },
    ]

    reviewed, report = finalize_translation_text(text, glossary_terms=terms)

    assert reviewed.startswith("# Capítulo 1: Após o Combate Mortal")
    assert "SOGOU" not in reviewed
    assert "SERAS" not in reviewed
    assert "Sogou viu Seras" in reviewed
    assert report["editorial"]["all_caps_name_replacements"]


def test_finalize_review_normalizes_caps_name_parts_and_creatures() -> None:
    terms = [
        {
            "key": "Kirihara Takuto",
            "pt": "Kirihara Takuto",
            "category": "personagem",
        },
        {
            "key": "Piggymaru",
            "pt": "Piggymaru",
            "category": "criatura",
        },
    ]

    reviewed, _ = finalize_translation_text("KIRIHARA falou com PIGGYMARU.", glossary_terms=terms)

    assert reviewed == "Kirihara falou com Piggymaru."


def test_finalize_review_does_not_normalize_common_word_from_translated_title() -> None:
    terms = [
        {
            "key": "Wildly Beautiful Emperor’s Elder Brother",
            "pt": "Irmão Mais Velho do Imperador Selvagem e Belo",
            "category": "personagem",
        }
    ]

    reviewed, _ = finalize_translation_text("“ANTES DE MAIS NADA—vamos.”", glossary_terms=terms)

    assert "MAIS" in reviewed


def test_review_normalizes_case_only_forbidden_target_alias() -> None:
    terms = [
        {
            "key": "Kyokugen",
            "pt": "Kyokugen",
            "category": "técnica",
            "bad_aliases": ["kyokugen"],
        }
    ]

    reviewed, report = review_translation_text("A habilidade de kyokugen pesa.", glossary_terms=terms)

    assert reviewed == "A habilidade de Kyokugen pesa."
    assert report.glossary_replacements == {"kyokugen->Kyokugen": 1}


def test_finalize_review_closes_one_dangling_curly_quote() -> None:
    reviewed, report = finalize_translation_text("“Primeira fala sem fechamento.\n\n“Segunda fala.”")

    assert reviewed == "“Primeira fala sem fechamento.”\n\n“Segunda fala.”"
    assert report["quote_balance_fixed"] is True


def test_finalize_review_restores_one_missing_curly_open_quote() -> None:
    reviewed, report = finalize_translation_text("Fala sem abertura.”\n\n“Outra fala.”")

    assert reviewed == "“Fala sem abertura.”\n\n“Outra fala.”"
    assert report["quote_balance_fixed"] is True


def test_finalize_review_collapses_blank_line_inside_same_quote() -> None:
    reviewed, report = finalize_translation_text("“Primeira parte.\n\nContinuação da mesma fala.”")

    assert reviewed == "“Primeira parte. Continuação da mesma fala.”"
    assert report["quote_blank_lines_fixed"] == 1


def test_finalize_review_normalizes_straight_quotes_and_missing_spacing() -> None:
    text = '"C-certamente…" Ela respirou. "Ehm… obrigada."'

    reviewed, report = finalize_translation_text(text)

    assert reviewed == "“C-certamente…” Ela respirou. “Ehm… obrigada.”"
    assert '"' not in reviewed
    assert report["quality"]["issues_by_type"].get("missing_quote_spacing") is None
