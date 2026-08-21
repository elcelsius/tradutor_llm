from tradutor.structure_normalizer import normalize_structure


def test_heading_with_inline_text_is_split():
    """Processamento interno auxiliar."""
    text = "Prólogo SOGOU AYAKA PAROU O GOLPE.\n\nOutro parágrafo."
    result = normalize_structure(text)
    assert "Prólogo" in result
    lines = [ln for ln in result.splitlines() if ln.strip()]
    assert lines[0] == "Prólogo"
    assert lines[1] == "SOGOU AYAKA PAROU O GOLPE."


def test_idempotent_normalize_structure():
    """Processamento interno auxiliar."""
    text = "Prólogo\n\nSOGOU AYAKA PAROU O GOLPE.\n\nOutro parágrafo."
    once = normalize_structure(text)
    twice = normalize_structure(once)
    assert once == twice


def test_chapter_title_glued_to_first_sentence_is_split():
    """Processamento interno auxiliar."""
    text = "# Capítulo 1:\n\nDepois do Deathmatch DEPOIS QUE SOGOU viu o corpo do Kirihara congelar."
    result = normalize_structure(text)

    assert result.startswith("# Capítulo 1: Após o Combate Mortal")
    assert "Depois que Sogou viu o corpo do Kirihara congelar." in result
    assert "Deathmatch DEPOIS" not in result


def test_missing_chapter_heading_for_deathmatch_opening_is_restored():
    """Processamento interno auxiliar."""
    text = "Após o combate mortal, após Sogou ter visto o corpo do Kirihara congelar, ela perdeu os sentidos."
    result = normalize_structure(text)

    assert result.startswith("# Capítulo 1: Após o Combate Mortal")
    assert "Após Sogou ter visto o corpo do Kirihara congelar" in result


def test_chapter_subtitle_is_merged_into_markdown_heading():
    """Processamento interno auxiliar."""
    text = "# Capítulo 1:\n\nApós o Combate Mortal\n\nDepois que Sogou acordou."

    result = normalize_structure(text)

    assert result.startswith("# Capítulo 1: Após o Combate Mortal")
    assert "Depois que Sogou acordou." in result


def test_chapter_subtitle_before_restored_heading_is_merged() -> None:
    """Recupera a ordem quando o heading foi inserido após o subtítulo."""
    text = "Após o Combate Mortal\n\n# Capítulo 1:\n\nDepois que Sogou acordou."

    result = normalize_structure(text)

    assert result.startswith("# Capítulo 1: Após o Combate Mortal")
    assert result.count("# Capítulo 1:") == 1
    assert "Depois que Sogou acordou." in result


def test_markdown_chapter_subtitle_is_merged_into_heading() -> None:
    text = "# Capítulo 2:\n\n## Vontades Turbulentas I\n\nDesci da carroça."

    result = normalize_structure(text)

    assert result.startswith("# Capítulo 2: Vontades Turbulentas I")
    assert "## Vontades" not in result
    assert "Desci da carroça." in result


def test_one_word_markdown_chapter_subtitle_is_merged_into_heading() -> None:
    text = "# Capítulo 5:\n\n## Conexões\n\nMinha consciência voltou à tona."

    result = normalize_structure(text)

    assert result.startswith("# Capítulo 5: Conexões")
    assert "Minha consciência voltou à tona." in result


def test_markdown_subtitle_beats_preceding_status_line() -> None:
    text = (
        "Mimori Touka Nível 5999\n\n"
        "Título: Herói de Classe E\n\n"
        "# Capítulo 5:\n\n"
        "## Conexões\n\n"
        "Minha consciência voltou à tona."
    )

    result = normalize_structure(text)

    assert "Título: Herói de Classe E" in result
    assert "# Capítulo 5: Conexões" in result
    assert "# Capítulo 5: Título" not in result
    assert "## Conexões" not in result


def test_duplicate_generic_heading_is_removed_after_a_titled_heading():
    """Processamento interno auxiliar."""
    text = "# Capítulo 1: Após a Batalha de Morte\n\n# Capítulo 1:\n\nDEPOIS QUE SOGOU viu Seras."

    result = normalize_structure(text)

    assert result.count("# Capítulo 1:") == 1
    assert "# Capítulo 1: Após a Batalha de Morte" in result
    assert "Depois que Sogou viu Seras." in result


def test_character_time_label_is_split_into_structure():
    """Processamento interno auxiliar."""
    text = "Yasu Tomohiro ALGUM TEMPO ANTES, há um tempo…"

    result = normalize_structure(text)

    assert result == "## Yasu Tomohiro\n\nAlgum tempo antes…"


def test_scene_separator_glued_to_narration_is_isolated():
    """Processamento interno auxiliar."""
    result = normalize_structure("*** A cena continua.")

    assert result == "***\n\nA cena continua."
