from tradutor.cleanup import cleanup_before_refine


def test_dedupe_prefix_lines_removes_truncated_and_is_idempotent():
    text = (
        "A habilidade única do Kirihara   agora estava no nível 3, e ele\n"
        "A habilidade única do Kirihara agora estava no nível 3, e ele havia aprendido...\n"
        "Linha normal."
    )

    cleaned, stats = cleanup_before_refine(text)
    assert "havia aprendido" in cleaned
    assert cleaned.count("A habilidade única do Kirihara") == 1
    assert stats["prefix_lines_removed"] == 1

    cleaned2, stats2 = cleanup_before_refine(cleaned)
    assert cleaned2 == cleaned
    assert stats2["prefix_lines_removed"] == 0
