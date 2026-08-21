from tradutor.postprocess_translation import postprocess_translation
from tradutor.qa import needs_retry
from tradutor.translate import _normalize_chunk_dialogue_quotes


def test_needs_retry_unbalanced_curly_quotes():
    """Processamento interno auxiliar."""
    input_text = "“How did you get there?”"
    output_text = "“Como você chegou lá?"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is True
    assert "unbalanced_quotes" in reason


def test_chunk_quote_normalizer_removes_one_spurious_terminal_straight_quote() -> None:
    """Processamento interno auxiliar."""
    source = "“The dialogue ends here.” The narration continues."
    translated = '"A fala termina aqui." A narração continua."'

    normalized = _normalize_chunk_dialogue_quotes(source, translated)

    assert normalized == "“A fala termina aqui.” A narração continua."


def test_chunk_quote_normalizer_removes_premature_curly_close_before_laughter() -> None:
    """Processamento interno auxiliar."""
    source = "Vicius riu. “Speak properly, won't you? Pfft, hee hee! Pathetic!”"
    translated = "Vicius riu. “Fale direito, pode ser?” Pfft, hee hee! Patética!”"

    normalized = _normalize_chunk_dialogue_quotes(source, translated)

    assert (
        normalized == "Vicius riu. “Fale direito, pode ser? Pfft, hee hee! Patética!”"
    )


def test_chunk_quote_normalizer_removes_spurious_terminal_narration_pair() -> None:
    """Não converte o último parágrafo narrativo da fonte em diálogo."""
    source = "“A fala anterior terminou.”\n\nA narração final continua sem aspas."
    translated = "“A fala anterior terminou.”\n\n“A narração final continua sem aspas.”"

    normalized = _normalize_chunk_dialogue_quotes(source, translated)

    assert normalized == "“A fala anterior terminou.”\n\nA narração final continua sem aspas."
    retry, reason = needs_retry(source, normalized)
    assert retry is False
    assert reason == ""


def test_chunk_quote_normalizer_removes_terminal_inline_narration_pair() -> None:
    source = "“The dialogue ended.”\nThe narrator reflects on what happened."
    translated = (
        "“A fala terminou.”\n\n"
        "O narrador reflete sobre o que aconteceu. “Acho que entendi.”"
    )

    normalized = _normalize_chunk_dialogue_quotes(source, translated)

    assert normalized == (
        "“A fala terminou.”\n\n"
        "O narrador reflete sobre o que aconteceu. Acho que entendi."
    )
    retry, reason = needs_retry(source, normalized)
    assert retry is False
    assert reason == ""


def test_chunk_quote_normalizer_removes_terminal_unmatched_close_from_narration() -> None:
    source = "“The dialogue ended.”\n\nThe final narration continues without quotes."
    translated = "“A fala terminou.”\n\nA narração final continua sem aspas.”"

    normalized = _normalize_chunk_dialogue_quotes(source, translated)

    assert normalized == "“A fala terminou.”\n\nA narração final continua sem aspas."
    retry, reason = needs_retry(source, normalized)
    assert retry is False
    assert reason == ""


def test_chunk_quote_normalizer_removes_spurious_inline_sound_pair() -> None:
    """Uma onomatopeia narrativa não deve aumentar a contagem de falas."""
    source = "She jumped down with a hup.\n\n“Are you all right?”"
    translated = "Ela pulou com um “hup”.\n\n“Você está bem?”"

    normalized = _normalize_chunk_dialogue_quotes(source, translated)

    assert normalized == "Ela pulou com um hup.\n\n“Você está bem?”"
    retry, reason = needs_retry(source, normalized)
    assert retry is False
    assert reason == ""


def test_needs_retry_allows_quote_boundary_in_source_chunk() -> None:
    """Processamento interno auxiliar."""
    input_text = "From my perspective, this can be resolved.”"
    output_text = "Do meu ponto de vista, isso pode ser resolvido.”"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is False
    assert reason == ""


def test_needs_retry_allows_odd_quote_boundary_when_style_changes() -> None:
    """Processamento interno auxiliar."""
    input_text = "“First.”\n\n“Second.”\n\nThird?”"
    output_text = '"Primeiro."\n\n"Segundo."\n\n"Terceiro'

    retry, reason = needs_retry(input_text, output_text)

    assert retry is False
    assert reason == ""


def test_needs_retry_rejects_extra_close_against_source_boundary() -> None:
    """Processamento interno auxiliar."""
    input_text = "From my perspective, this can be resolved.”"
    output_text = "Do meu ponto de vista, isso pode ser resolvido.””"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is True
    assert reason == "unbalanced_quotes"


def test_needs_retry_rejects_extra_balanced_quote_pair() -> None:
    """Processamento interno auxiliar."""
    input_text = "“Primeira fala.”"
    output_text = "“Primeira fala.”\n\n“Fala inventada.”"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is True
    assert reason == "extra_curly_quotes"


def test_needs_retry_allows_one_quote_pair_that_repairs_internal_source_defect() -> (
    None
):
    """Processamento interno auxiliar."""
    input_text = (
        "“First speech.”\n\n"
        "From my perspective, this can be resolved.”\n\n"
        "“No, forget it."
    )
    output_text = (
        "“Primeira fala.”\n\n"
        "“Do meu ponto de vista, isso pode ser resolvido.”\n\n"
        "“Não, esqueça.”"
    )

    retry, reason = needs_retry(input_text, output_text)

    assert retry is False
    assert reason == ""


def test_needs_retry_allows_single_missing_open_quote_repair() -> None:
    """Processamento interno auxiliar."""
    input_text = "I believe the class shares that intention.”"
    output_text = "“Acredito que a turma compartilhe essa intenção.”"

    retry, reason = needs_retry(input_text, output_text)

    assert retry is False
    assert reason == ""


def test_needs_retry_extra_short_repetition():
    """Processamento interno auxiliar."""
    input_text = "Crack!\nSilence."
    output_text = "Crack!\nCrack!\nCrack!\nSilence."

    retry, reason = needs_retry(input_text, output_text)

    assert retry is True
    assert "extra_short_repetition" in reason


def test_needs_retry_allows_short_line_repetition_present_in_source() -> None:
    """Processamento interno auxiliar."""
    input_text = "\n\n".join(["“On your knees.”"] * 5)
    output_text = "\n\n".join(["“Ajoelhe-se.”"] * 5)

    retry, reason = needs_retry(input_text, output_text)

    assert retry is False
    assert reason == ""


def test_dash_line_strips_trailing_quote():
    """Processamento interno auxiliar."""
    samples = [
        ("— Entendido”, Seras me respondeu.", "— Entendido, Seras me respondeu."),
        ("— Oh?!” Vicius perguntou.", "— Oh?! Vicius perguntou."),
        ("— Eh?” Vicius disse.", "— Eh? Vicius disse."),
        ("— Hm.” Seras comentou.", "— Hm. Seras comentou."),
        ("— …” ele murmurou.", "— … ele murmurou."),
    ]

    for raw, expected in samples:
        cleaned = postprocess_translation(
            raw, en_text=""
        )  # en_text vazio para passar pelo pipeline
        assert cleaned == expected


def test_postprocess_translation_collapses_duplicate_curly_quote() -> None:
    """Processamento interno auxiliar."""
    assert (
        postprocess_translation("A fala terminou.””", en_text="") == "A fala terminou.”"
    )


def test_postprocess_translation_removes_straight_quote_adjacent_to_curly_quote() -> None:
    cleaned = postprocess_translation(
        '“Você só estava cansado depois da luta, não foi?” "', en_text=""
    )

    assert cleaned == "“Você só estava cansado depois da luta, não foi?”"


def test_postprocess_merges_lowercase_continuation_after_ellipsis() -> None:
    """Uma quebra de parágrafo artificial não deve iniciar narração em minúscula."""
    source = "He returned after the Great Invasion…\nbut he sought treatment elsewhere."
    translated = (
        "Ele voltou após a Grande Invasão…\n\n"
        "mas buscou tratamento em outro lugar."
    )

    cleaned = postprocess_translation(translated, en_text=source)
    retry, reason = needs_retry(source, cleaned)

    assert cleaned == "Ele voltou após a Grande Invasão… mas buscou tratamento em outro lugar."
    assert retry is False
    assert reason == ""
