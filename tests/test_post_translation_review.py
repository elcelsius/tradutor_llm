from tradutor.post_translation_review import (
    finalize_translation_text,
    review_translation_text,
)
from tradutor.translate import ensure_section_heading, source_heading_to_pt


def test_source_heading_to_pt() -> None:
    """Processamento interno auxiliar."""
    assert source_heading_to_pt("Chapter 5:") == "# Capítulo 5:"
    assert source_heading_to_pt("Epilogue") == "# Epílogo"
    assert source_heading_to_pt("Afterword") == "# Pós-escrito"
    assert source_heading_to_pt("Full Text") is None


def test_ensure_section_heading_inserts_missing_heading() -> None:
    """Processamento interno auxiliar."""
    text, changed = ensure_section_heading(
        "Minha consciência voltou à tona.", "Chapter 5:"
    )

    assert changed is True
    assert text.startswith("# Capítulo 5:\n\nMinha consciência")


def test_ensure_section_heading_does_not_duplicate_heading() -> None:
    """Processamento interno auxiliar."""
    text, changed = ensure_section_heading("# Capítulo 2:\n\nTexto.", "Chapter 2:")

    assert changed is False
    assert text == "# Capítulo 2:\n\nTexto."


def test_review_does_not_restore_generic_heading_when_titled_heading_exists() -> None:
    """Processamento interno auxiliar."""
    text = "# Capítulo 1: Após o Combate Mortal\n\nDepois que Sogou acordou."

    reviewed, report = review_translation_text(text, sections=[{"title": "Chapter 1:"}])

    assert reviewed.count("# Capítulo 1:") == 1
    assert report.heading_fixes == 0


def test_review_recognizes_unaccented_equivalent_heading() -> None:
    text = "# Prologo\n\nTexto do prólogo."

    reviewed, report = review_translation_text(text, sections=[{"title": "Prologue"}])

    assert reviewed.count("#") == 1
    assert reviewed.startswith("# Prólogo")
    assert report.heading_fixes == 1


def test_finalize_merges_restored_heading_with_leading_chapter_subtitle() -> None:
    """Preserva o subtitulo quando a LLM omite o heading do capitulo."""
    reviewed, report = finalize_translation_text(
        "Após o Combate Mortal\n\nDepois que Sogou acordou.",
        sections=[{"title": "Chapter 1:"}],
    )

    assert reviewed.startswith("# Capítulo 1: Após o Combate Mortal")
    assert reviewed.count("# Capítulo 1:") == 1
    assert report["editorial"]["heading_fixes"] == 1


def test_finalize_preserves_status_before_markdown_chapter_subtitle() -> None:
    text = (
        "Mimori Touka Nível 5999\n\n"
        "Título: Herói do Título E-class de Touka\n\n"
        "# Capítulo 5:\n\n"
        "## Conexões\n\n"
        "Minha consciência voltou à tona."
    )
    terms = [
        {
            "key": "E-Class Hero",
            "pt": "Herói de Classe E",
            "bad_aliases": ["Herói do Título E-class de Touka"],
        }
    ]

    reviewed, _ = finalize_translation_text(text, glossary_terms=terms)

    assert "Título: Herói de Classe E" in reviewed
    assert "# Capítulo 5: Conexões" in reviewed
    assert "# Capítulo 5: Título" not in reviewed
    assert "## Conexões" not in reviewed


def test_review_translation_restores_headings_and_applies_alias_fixes() -> None:
    """Processamento interno auxiliar."""
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
        },
    ]

    reviewed, report = review_translation_text(
        text, sections=sections, glossary_terms=terms
    )

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
    reviewed_male, _ = review_translation_text(
        "A Mimori como aliada.", glossary_terms=terms
    )
    assert reviewed_male == "O Mimori como aliado."
    assert "manipulando a Asagi" in reviewed
    assert "Filhos de Vicius" in reviewed
    assert report.heading_fixes == 2
    assert report.glossary_replacements["Discípulos de Vicius->Filhos de Vicius"] == 1


def test_review_fixes_gendered_articles_for_feminine_creatures() -> None:
    """Processamento interno auxiliar."""
    terms = [
        {"key": "Slei", "pt": "Slei", "category": "criatura", "gender": "feminino"}
    ]

    reviewed, _ = review_translation_text(
        "O Slei avançou nas costas do Slei.", glossary_terms=terms
    )

    assert reviewed == "A Slei avançou nas costas da Slei."


def test_review_fixes_plural_articles_for_feminine_monsters() -> None:
    terms = [
        {
            "key": "Eucharists",
            "pt": "eucaristias",
            "category": "monstro",
            "gender": "feminino",
        }
    ]

    reviewed, _ = review_translation_text(
        "Os eucaristias avançaram contra os heróis.", glossary_terms=terms
    )

    assert reviewed == "As eucaristias avançaram contra os heróis."


def test_review_applies_contextual_glossary_target_replacement() -> None:
    """Processamento interno auxiliar."""
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
    assert (
        report.glossary_replacements["Após a Batalha de Morte->Após o Combate Mortal"]
        == 1
    )


def test_review_prefers_longest_overlapping_forbidden_alias() -> None:
    terms = [
        {
            "key": "Touka E-class Title",
            "pt": "título de classe E de Touka",
            "bad_aliases": ["Título E-class de Touka"],
        },
        {
            "key": "E-Class Hero",
            "pt": "Herói de Classe E",
            "bad_aliases": ["Herói do Título E-class de Touka"],
        },
    ]

    reviewed, report = review_translation_text(
        "Título: Herói do Título E-class de Touka", glossary_terms=terms
    )

    assert reviewed == "Título: Herói de Classe E"
    assert report.glossary_replacements == {
        "Herói do Título E-class de Touka->Herói de Classe E": 1
    }


def test_review_corrects_gendered_heading_alias_from_glossary() -> None:
    terms = [
        {
            "key": "The White Goddess and the Traitor",
            "pt": "A Deusa Branca e a Traidora",
            "bad_aliases": ["A Deusa Branca e o Traidor"],
        }
    ]

    reviewed, _ = review_translation_text(
        "# Capítulo 4: A Deusa Branca e o Traidor", glossary_terms=terms
    )

    assert reviewed == "# Capítulo 4: A Deusa Branca e a Traidora"


def test_review_leaves_contextual_alias_for_chunk_repair() -> None:
    terms = [
        {
            "key": "Eucharists",
            "pt": "eucaristias",
            "contextual_bad_aliases": ["eucaristos"],
        }
    ]

    reviewed, report = review_translation_text(
        "Os eucaristos gigantes avançaram.", glossary_terms=terms
    )

    assert reviewed == "Os eucaristos gigantes avançaram."
    assert report.glossary_replacements == {}


def test_review_collapses_duplicate_canonical_character_names() -> None:
    """Processamento interno auxiliar."""
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
    """Processamento interno auxiliar."""
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
    """Processamento interno auxiliar."""
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

    reviewed, _ = finalize_translation_text(
        "KIRIHARA falou com PIGGYMARU.", glossary_terms=terms
    )

    assert reviewed == "Kirihara falou com Piggymaru."


def test_finalize_review_does_not_normalize_common_word_from_translated_title() -> None:
    """Processamento interno auxiliar."""
    terms = [
        {
            "key": "Wildly Beautiful Emperor’s Elder Brother",
            "pt": "Irmão Mais Velho do Imperador Selvagem e Belo",
            "category": "personagem",
        }
    ]

    reviewed, _ = finalize_translation_text(
        "“ANTES DE MAIS NADA—vamos.”", glossary_terms=terms
    )

    assert "MAIS" in reviewed


def test_review_normalizes_case_only_forbidden_target_alias() -> None:
    """Processamento interno auxiliar."""
    terms = [
        {
            "key": "Kyokugen",
            "pt": "Kyokugen",
            "category": "técnica",
            "bad_aliases": ["kyokugen"],
        }
    ]

    reviewed, report = review_translation_text(
        "A habilidade de kyokugen pesa.", glossary_terms=terms
    )

    assert reviewed == "A habilidade de Kyokugen pesa."
    assert report.glossary_replacements == {"kyokugen->Kyokugen": 1}


def test_finalize_review_closes_one_dangling_curly_quote() -> None:
    """Processamento interno auxiliar."""
    reviewed, report = finalize_translation_text(
        "“Primeira fala sem fechamento.\n\n“Segunda fala.”"
    )

    assert reviewed == "“Primeira fala sem fechamento.”\n\n“Segunda fala.”"
    assert report["quote_balance_fixed"] is True


def test_finalize_review_restores_one_missing_curly_open_quote() -> None:
    """Processamento interno auxiliar."""
    reviewed, report = finalize_translation_text("Fala sem abertura.”\n\n“Outra fala.”")

    assert reviewed == "“Fala sem abertura.”\n\n“Outra fala.”"
    assert report["quote_balance_fixed"] is True


def test_finalize_review_repairs_known_recovery_grammar_defects() -> None:
    """A revisão final corrige erros recorrentes que não exigem julgamento de estilo."""
    reviewed, report = finalize_translation_text(
        "Assim que ela recuperar-se da consciência, vamos rezar para que ela se recuperar. "
        "Ele parece estar se recuperando-se mais rápido."
    )

    assert "recuperar a consciência" in reviewed
    assert "para que ela se recupere" in reviewed
    assert "se recuperando mais rápido" in reviewed
    assert report["quality"]["issue_count"] == 0


def test_finalize_review_does_not_singularize_unrelated_plural_consciousness() -> None:
    text = "Eles tentaram recuperar-se das consciências de seus eus passados."

    reviewed, report = finalize_translation_text(text)

    assert reviewed == text
    assert report["editorial"]["text_replacements"].get(
        "recover_consciousness", 0
    ) == 0


def test_finalize_review_collapses_blank_line_inside_same_quote() -> None:
    """Processamento interno auxiliar."""
    reviewed, report = finalize_translation_text(
        "“Primeira parte.\n\nContinuação da mesma fala.”"
    )

    assert reviewed == "“Primeira parte. Continuação da mesma fala.”"
    assert report["quote_blank_lines_fixed"] == 1


def test_finalize_review_normalizes_straight_quotes_and_missing_spacing() -> None:
    """Processamento interno auxiliar."""
    text = '"C-certamente…" Ela respirou. "Ehm… obrigada."'

    reviewed, report = finalize_translation_text(text)

    assert reviewed == "“C-certamente…” Ela respirou. “Ehm… obrigada.”"
    assert '"' not in reviewed
    assert report["quality"]["issues_by_type"].get("missing_quote_spacing") is None


def test_review_applies_high_confidence_pt_br_grammar_fixes() -> None:
    text = (
        "Pode ser uma bênção alguém do seu nível ter sido mandado. "
        "Você vai assistir os seus amados serem levados."
    )

    reviewed, report = review_translation_text(text)

    assert "Pode ser uma bênção que alguém do seu nível tenha sido mandado." in reviewed
    assert "assistir aos seus amados serem levados" in reviewed
    assert report.text_replacements["predicative_subordinate_connector"] == 1
    assert report.text_replacements["assistir_aos_passive"] == 1


def test_review_repairs_known_brainwash_and_dialogue_calques() -> None:
    text = (
        "Sogou foi lavada cerebral pela Deusa. "
        "Itsuki foi lavada o cérebro. "
        "Pode ter sido lavada cerebral antes. "
        "Sua pureza facilita para os outros tingi-la. "
        "Ela facilita que os outros a tingem com suas cores. "
        "Eu tô meio que sentindo muito, sabe?"
    )

    reviewed, report = review_translation_text(text)

    assert "Sogou sofreu lavagem cerebral pela Deusa." in reviewed
    assert "Itsuki teve o cérebro lavado." in reviewed
    assert "ter sofrido lavagem cerebral antes." in reviewed
    assert "facilita que os outros a tinjam." in reviewed
    assert "facilita que os outros a tinjam com suas cores." in reviewed
    assert "Eu meio que sinto muito, sabe?" in reviewed
    assert report.text_replacements["brainwash_literal_participle"] == 1
    assert report.text_replacements["brainwash_literal_brain"] == 1
    assert report.text_replacements["brainwash_literal_infinitive"] == 1
    assert report.text_replacements["facilitar_tingir_subjunctive"] == 2
    assert report.text_replacements["apology_progressive"] == 1


def test_review_repairs_observed_literary_calques_conservatively() -> None:
    text = (
        "Os olhos de Munin baixaram para as palmas de suas mãos. "
        "Não sou falho ou perfeito. "
        "A régua, o que a torna difícil de confiar, não ajuda. "
        "Nyaki é uma membro importante."
    )

    reviewed, report = review_translation_text(text)

    assert "Munin baixou os olhos para as palmas das mãos." in reviewed
    assert "Não sou impecável nem perfeito." in reviewed
    assert "o que torna difícil confiar nela" in reviewed
    assert "Nyaki é uma integrante importante." in reviewed
    assert report.text_replacements["eyes_dropped"] == 1
    assert report.text_replacements["not_flawless"] == 1
    assert report.text_replacements["hard_to_trust"] == 1
    assert report.text_replacements["feminine_member"] == 1


def test_review_repairs_observed_contextual_grammar_artifacts() -> None:
    text = (
        "Desde que Vicius não o capture e sofreu lavagem cerebral nele. "
        "A régua que a gente usa pra julgar ela que faz ela parecer difícil de confiar. "
        "Ela estava na pose de ponte de ginástica. "
        "Pelo seu expressão, ela hesitou."
    )

    reviewed, report = review_translation_text(text)

    assert "não o capture nem o submeta a uma lavagem cerebral" in reviewed
    assert "com que a gente a julga que torna difícil confiar nela" in reviewed
    assert "na posição de ponte de ginástica" in reviewed
    assert "Pela sua expressão, ela hesitou." in reviewed
    assert report.text_replacements["capture_brainwash"] == 1
    assert report.text_replacements["hard_to_trust"] == 1
    assert report.text_replacements["bridge_pose"] == 1
    assert report.text_replacements["possessive_expression"] == 1


def test_review_repairs_observed_volume_11_literary_artifacts() -> None:
    text = (
        "as pessoas são congeladas quando um for desenvolvido.. "
        "Seras e Munin estavam tendo uma conversa um pouco afastado. "
        "quantos dos nossos colegas ainda são considerados como estando em Alion no momento. "
        "há um risco de que ele tenha sofrido lavagem cerebral e manipulado. "
        "Ser atencioso… É isso que a Hijiri culpa pelo seu fracasso. "
        "Não era ela que estava sentada dois assentos ao meu lado? "
        "A menina pequena, Yuri, agarrou-se à mãe. "
        "Dei um grunhido de deboche. "
        "você e os outros membros do 2-C pensam muito bem de mim. "
        "Todo mundo pensa muito bem de mim."
    )

    reviewed, report = review_translation_text(text)

    assert "quando um tratamento for desenvolvido." in reviewed
    assert "estavam conversando um pouco mais longe." in reviewed
    assert "quantos dos nossos colegas ainda devem estar em Alion." in reviewed
    assert "tenha sofrido lavagem cerebral e sido manipulado." in reviewed
    assert "É por isso que Hijiri se culpa pelo próprio fracasso." in reviewed
    assert "sentada a dois assentos de distância de mim?" in reviewed
    assert "A pequena Yuri agarrou-se à mãe." in reviewed
    assert "Bufei para ela." in reviewed
    assert "têm uma opinião elevada demais sobre mim." in reviewed
    assert "Todo mundo tem uma opinião elevada demais sobre mim." in reviewed
    assert report.text_replacements["missing_treatment_noun"] == 1
    assert report.text_replacements["chatting_away"] == 1
    assert report.text_replacements["classmates_location"] == 1
    assert report.text_replacements["brainwash_and_manipulated"] == 1
    assert report.text_replacements["self_blame"] == 1
    assert report.text_replacements["two_seats_down"] == 1
    assert report.text_replacements["little_girl_name"] == 1
    assert report.text_replacements["snorted_at_her"] == 1
    assert report.text_replacements["thinks_too_highly"] == 2


def test_review_repairs_real_you_and_self_blame_variants() -> None:
    text = (
        "Acredito que o verdadeiro você está mais adaptado para sobreviver neste mundo. "
        "Acredito que o verdadeiro você seja mais adequado para sobreviver nesse mundo. "
        "E devo adicionar — isso é relevante. "
        "Isso é o que a Hijiri culpa pelo seu fracasso. "
        "Esse é o tipo de falha de que podemos recuperar. "
        "Seras e Munin estavam tendo uma conversa um pouco distante."
    )

    reviewed, report = review_translation_text(text)

    assert "Acredito que o seu verdadeiro eu esteja mais apto a sobreviver neste mundo." in reviewed
    assert reviewed.count("Acredito que o seu verdadeiro eu esteja mais apto a sobreviver neste mundo.") == 2
    assert "E devo acrescentar — isso é relevante." in reviewed
    assert "É por isso que Hijiri se culpa pelo próprio fracasso." in reviewed
    assert "Esse é o tipo de falha de que podemos nos recuperar." in reviewed
    assert "estavam conversando um pouco mais longe." in reviewed
    assert report.text_replacements["real_you"] == 2
    assert report.text_replacements["must_add"] == 1
    assert report.text_replacements["self_blame"] == 1
    assert report.text_replacements["recover_from"] == 1
    assert report.text_replacements["chatting_away"] == 1


def test_review_repairs_second_pass_literary_variants() -> None:
    text = (
        "Seras e Munin estavam tendo uma conversa um pouco mais afastadas. "
        "Acredito que o verdadeiro você esteja mais adaptado a sobreviver neste mundo. "
        "você e os outros membros da 2-C pensam demais em mim. "
        "Todos pensam demais de mim. "
        "Esse é o tipo de falha que podemos recuperar. "
        "Ela tem sido longe de inativa. "
        "Fácil de sofrer lavagem cerebral, em outras palavras. "
        "Só se alguém fosse sofreu lavagem cerebral que conseguiria confiar nela. "
        "o critério que faz com que seja difícil confiar nela. "
        "Você realmente tentando me vencer? "
        "Nyaki está tão grata por ter vindo! "
        "O choque mental pode levar um tempo para ela se recuperar de. "
        "Sogou e Seras deveriam ter se juntado juntos. "
        "Na mão direita de Vicius, ela segurava a cabeça decepada de Lokiella."
    )

    reviewed, report = review_translation_text(text)

    assert "estavam conversando um pouco mais longe." in reviewed
    assert "Acredito que o seu verdadeiro eu esteja mais apto a sobreviver neste mundo." in reviewed
    assert "têm uma opinião elevada demais sobre mim." in reviewed
    assert "você e os outros membros da 2-C têm uma opinião elevada demais sobre mim." in reviewed
    assert "Todos têm uma opinião elevada demais sobre mim." in reviewed
    assert "Esse é o tipo de falha de que podemos nos recuperar." in reviewed
    assert "Ela tem estado bem ativa." in reviewed
    assert "Em outras palavras, isso a torna vulnerável à lavagem cerebral." in reviewed
    assert "Só alguém que tivesse sofrido lavagem cerebral conseguiria confiar nela." in reviewed
    assert "o critério que torna difícil confiar nela." in reviewed
    assert "Você realmente estava tentando me vencer?" in reviewed
    assert "Nyaki está tão contente por ter vindo!" in reviewed
    assert "O choque mental pode levar algum tempo para ser superado." in reviewed
    assert "Sogou e Seras deveriam ter se juntado." in reviewed
    assert "Vicius segurava a cabeça decepada de Lokiella na mão direita." in reviewed
    assert report.text_replacements["chatting_away"] == 1
    assert report.text_replacements["real_you"] == 1
    assert report.text_replacements["thinks_too_highly"] == 2
    assert report.text_replacements["recover_from"] == 1
    assert report.text_replacements["been_far_from_inactive"] == 1
    assert report.text_replacements["easy_to_brainwash"] == 1
    assert report.text_replacements["only_brainwashed_can_trust"] == 1
    assert report.text_replacements["hard_to_trust"] == 1
    assert report.text_replacements["missing_auxiliary_beat"] == 1
    assert report.text_replacements["glad_came"] == 1
    assert report.text_replacements["mental_shock_recovery"] == 1
    assert report.text_replacements["duplicate_joined_together"] == 1
    assert report.text_replacements["right_hand_possessor"] == 1


def test_review_keeps_mental_shock_lowercase_inside_subordinate_clause() -> None:
    reviewed, report = review_translation_text(
        "Hijiri mencionou que o choque mental pode levar um tempo para ela se recuperar de."
    )

    assert (
        reviewed
        == "Hijiri mencionou que o choque mental pode levar algum tempo para ser superado."
    )
    assert report.text_replacements["mental_shock_recovery"] == 1
