from tradutor.quote_fix import fix_dialogue_artifacts


def test_falas_coladas_com_espaco():
    text = "“A!” “B!”"
    fixed = fix_dialogue_artifacts(text)
    assert fixed == "“A!”\n\n“B!”"


def test_falas_coladas_sem_espaco():
    text = "“A!”“B!”"
    fixed = fix_dialogue_artifacts(text)
    assert fixed == "“A!”\n\n“B!”"


def test_aspas_duplicadas_no_meio():
    text = "Algo aconteceu ”“Quando ele chegou."
    fixed = fix_dialogue_artifacts(text)
    assert "”\n\n“Quando" in fixed


def test_linha_em_branco_dentro_de_fala():
    text = "“Entendo.\n\nQuer dizer que sim.”"
    fixed = fix_dialogue_artifacts(text)
    assert "\n\n" not in fixed
    assert "“Entendo.\nQuer dizer que sim.”" == fixed


def test_paragrafo_fora_de_aspas_permanece():
    original = "Paragrafo A.\n\nParagrafo B."
    fixed = fix_dialogue_artifacts(original)
    assert fixed == original
