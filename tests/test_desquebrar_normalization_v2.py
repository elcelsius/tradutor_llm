from tradutor.desquebrar import (
    normalize_hardwrap_joins,
    normalize_internal_hyphen_by_dominance,
    normalize_scene_separators,
)


def test_hardwrap_join():
    text = "... voice, which\nmysteriously came ..."

    normalized, joins = normalize_hardwrap_joins(text)

    assert "which mysteriously" in normalized
    assert joins == 1


def test_internal_hyphen_dominance_applies():
    text = "understand\nunderstand\nunderstand\nunder-stand"

    normalized, stats = normalize_internal_hyphen_by_dominance(text)

    assert "under-stand" not in normalized
    assert "understand" in normalized
    assert stats.get("total", 0) >= 1


def test_internal_hyphen_keeps_legit_compound():
    text = "demi-humans are here.\nDemihumans are rare."

    normalized, stats = normalize_internal_hyphen_by_dominance(text)

    assert "demi-humans" in normalized
    assert stats.get("total", 0) == 0


def test_internal_hyphen_keeps_honorific():
    text = "Zine-sama greeted everyone."

    normalized, stats = normalize_internal_hyphen_by_dominance(text)

    assert "Zine-sama" in normalized
    assert stats.get("total", 0) == 0


def test_scene_separator_isolated():
    text = "Line A\n***\nLine B"

    normalized, fixes = normalize_scene_separators(text)

    assert "\n\n***\n\n" in normalized
    assert fixes >= 1
