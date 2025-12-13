from tradutor.cleanup import dedupe_adjacent_lines, fix_glued_dialogues, cleanup_before_refine


def test_dedupe_adjacent_lines_removes_repeats() -> None:
    src = "Linha A\nLinha A\n\nLinha B  \nLinha   B\nLinha B\n\n# Heading\n# Heading\n"
    result, stats = dedupe_adjacent_lines(src)
    assert result.count("Linha A") == 1
    assert result.splitlines().count("Linha B  ") + result.splitlines().count("Linha   B") == 1
    assert result.splitlines().count("# Heading") == 1
    assert stats["lines_removed"] >= 1
    assert stats["blocks_removed"] >= 0


def test_fix_glued_dialogues_inserts_newline() -> None:
    src = '"Oi." "Tudo bem?"'
    out, stats = fix_glued_dialogues(src)
    assert '"Oi."' in out
    assert '"Tudo bem?"' in out
    assert "\n" in out.strip()
    assert stats["breaks_inserted"] >= 1


def test_cleanup_idempotent() -> None:
    src = '"Oi." "Tudo bem?"\n\nLinha X\nLinha X\n\nParagrafo\nParagrafo\n'
    first, _ = cleanup_before_refine(src)
    second, stats2 = cleanup_before_refine(first)
    assert first == second
    assert stats2["lines_removed"] == 0
    assert stats2["blocks_removed"] == 0
    assert stats2["breaks_inserted"] == 0
