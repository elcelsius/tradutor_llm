from tradutor.quote_fix import fix_unbalanced_quotes, count_curly_quotes, fix_blank_lines_inside_quotes


def test_fix_unbalanced_quotes_inserts_closing_before_narration():
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
    balanced = "“Olá.” Ele disse. “Tchau.”"
    fixed, changed = fix_unbalanced_quotes(balanced, logger=None, label="test2")
    assert not changed
    assert fixed == balanced


def test_fix_blank_lines_inside_quotes_collapses():
    text = "“Ele falou algo.\n\nContinuou a frase.”\n\nFora do diálogo."
    cleaned, fixes = fix_blank_lines_inside_quotes(text, logger=None, label="blank")
    assert fixes == 1
    assert "algo.\nContinuou" in cleaned
    # Fora das aspas, parágrafo permanece
    assert "\n\nFora do diálogo." in cleaned
