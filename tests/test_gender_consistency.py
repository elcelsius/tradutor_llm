from tradutor.refine import build_refine_prompt


def test_gender_instruction_present() -> None:
    prompt = build_refine_prompt("dummy")
    lower = prompt.lower()
    assert "gênero" in lower
    assert "narrador" in lower
    assert "masculino" in lower
