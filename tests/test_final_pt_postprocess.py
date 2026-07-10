from tradutor.postprocess import final_pt_postprocess


def test_final_pt_postprocess_fixes_small_editorial_artifacts() -> None:
    text = (
        "O criatura caiu.””\n\n"
        "Eu fui mandada para as Ruínas do Descarte? Não: eu ter sido mandada para as Ruínas do Descarte.\n\n"
        "Precisamos enterrar o machado e reestrear contato.\n\n"
        "Foi sua multidão de estratégias que venceu o dia. Seu socorro e no combate foi essencial.\n\n"
        "A tenda de cortina ficava perto da parede de cortina."
    )

    result = final_pt_postprocess(text)

    assert "A criatura caiu.”" in result
    assert "””" not in result
    assert "eu ter sido mandado para as Ruínas do Descarte" in result
    assert "deixar isso para trás" in result
    assert "retomar contato" in result
    assert "muitas estratégias" in result
    assert "decidiu a batalha" in result
    assert "sua ajuda no combate" in result
    assert "cortina da tenda" in result
