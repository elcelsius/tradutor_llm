from tradutor.translate import build_translation_prompt


def test_plural_rule_present() -> None:
    p = build_translation_prompt("dummy")
    lower = p.lower()
    assert "plural" in lower
    assert "vocês" in lower  # regra identifica o termo proibido
