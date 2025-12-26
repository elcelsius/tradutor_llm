from tradutor.preprocess import remove_noise_blocks, strip_front_matter
from tradutor.preprocess import sanitize_extracted_text


def test_remove_noise_blocks_drops_ads() -> None:
    text = "Stay up to date with us.\n\nReal story starts here.\n\nJoin our Discord now!"
    cleaned = remove_noise_blocks(text)
    assert "Stay up to date" not in cleaned.lower()
    assert "join our discord" not in cleaned.lower()
    assert "Real story" in cleaned


def test_strip_front_matter_starts_at_prologue() -> None:
    text = "Title page\n\nTable of Contents\n\nPrologue\nLine 1\nLine 2"
    trimmed = strip_front_matter(text)
    assert trimmed.startswith("Prologue")
    assert "Title page" not in trimmed


def test_sanitize_extracted_removes_numeric_lines_and_uffff() -> None:
    raw = "Line 1\n2\nLine 3\n￿\n\n4\nReal line\n"
    cleaned, stats = sanitize_extracted_text(raw)
    assert "Line 1" in cleaned and "Line 3" in cleaned
    assert "\n2\n" not in cleaned
    assert "\n4\n" not in cleaned
    assert "￿" not in cleaned
    assert stats["removed_numeric_lines"] == 2
