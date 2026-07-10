from tradutor.post_translation_review import review_translation_text
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
    assert "manipulando a Asagi" in reviewed
    assert "Filhos de Vicius" in reviewed
    assert report.heading_fixes == 2
    assert report.glossary_replacements["Discípulos de Vicius->Filhos de Vicius"] == 1


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
