from tradutor.quote_fix import (
    count_curly_quotes,
    fix_blank_lines_inside_quotes,
    fix_unbalanced_quotes,
    repair_missing_closing_curly_quotes_per_paragraph,
    repair_missing_open_quotes_per_paragraph,
)


def test_fix_unbalanced_quotes_inserts_closing_before_narration():
    """Processamento interno auxiliar."""
    text = (
        "… “Essa coisa de cavaleiro é só para manter as aparências do reino, sabe? "
        "Ele bebeu um gole antes de continuar. “O Rei Caçador de Monstros…”"
    )
    fixed, changed = fix_unbalanced_quotes(text, logger=None, label="test")
    opens, closes = count_curly_quotes(fixed)
    assert changed
    assert opens == closes
    assert "sabe?” Ele bebeu" in fixed
    assert "O Rei Caçador de Monstros" in fixed


def test_fix_no_change_when_balanced():
    """Processamento interno auxiliar."""
    balanced = "“Olá.” Ele disse. “Tchau.”"
    fixed, changed = fix_unbalanced_quotes(balanced, logger=None, label="test2")
    assert not changed
    assert fixed == balanced


def test_fix_unbalanced_quotes_removes_stray_close_after_narration():
    """Remove um fechamento extra depois de uma fala ja encerrada."""
    text = (
        "“Os confins do norte? O que tem eles?” perguntou o imperador. "
        "Antes da luta, ele tentou trazer monstros para invadir Mira. "
        "Entre eles havia alguns que vieram dos confins do norte.”"
    )

    fixed, changed = fix_unbalanced_quotes(text, logger=None, label="stray-close")

    assert changed
    assert fixed.endswith("confins do norte.")
    assert count_curly_quotes(fixed) == (1, 1)


def test_repair_missing_open_quote_when_global_counts_are_misleading():
    """Processamento interno auxiliar."""
    text = "Fala que perdeu a abertura.”\n\n“Fala já íntegra.”"

    fixed, fixes = repair_missing_open_quotes_per_paragraph(
        text, logger=None, label="local"
    )

    assert fixes == 1
    assert fixed.startswith("“Fala que perdeu a abertura.”")
    assert count_curly_quotes(fixed) == (2, 2)


def test_repair_missing_open_quote_keeps_paired_inline_quote_unchanged():
    """Processamento interno auxiliar."""
    text = "A expressão “bem-vindo” continua corretamente delimitada."

    fixed, fixes = repair_missing_open_quotes_per_paragraph(
        text, logger=None, label="local"
    )

    assert fixes == 0
    assert fixed == text


def test_repair_missing_open_quote_accepts_isolated_music_note():
    """Uma fala musical curta pode perder só a abertura durante a extração."""
    fixed, fixes = repair_missing_open_quotes_per_paragraph(
        "♪”\n\nA narração continua.", logger=None, label="music"
    )

    assert fixes == 1
    assert fixed.startswith("“♪”")


def test_repair_missing_closing_curly_quote_at_paragraph_end():
    """Não deixa uma aspa reta terminal desequilibrar o chunk de tradução."""
    text = '“A fala preservou a abertura, mas não o fechamento."\n\nNarração.'

    fixed, fixes = repair_missing_closing_curly_quotes_per_paragraph(
        text, logger=None, label="terminal"
    )

    assert fixes == 1
    assert fixed.startswith("“A fala preservou a abertura, mas não o fechamento.”")
    assert count_curly_quotes(fixed) == (1, 1)


def test_repair_missing_closing_curly_quote_keeps_measurement_unchanged():
    """Aspas retas normais, como marca de polegadas, não são diálogo."""
    text = 'A placa mede 6" de largura.'

    fixed, fixes = repair_missing_closing_curly_quotes_per_paragraph(
        text, logger=None, label="measurement"
    )

    assert fixes == 0
    assert fixed == text


def test_fix_blank_lines_inside_quotes_collapses():
    """Processamento interno auxiliar."""
    text = "“Ele falou algo.\n\nContinuou a frase.”\n\nFora do diálogo."
    cleaned, fixes = fix_blank_lines_inside_quotes(text, logger=None, label="blank")
    assert fixes == 1
    assert "algo. Continuou" in cleaned
    # Fora das aspas, parágrafo permanece
    assert "\n\nFora do diálogo." in cleaned
