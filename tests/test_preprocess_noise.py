from tradutor.preprocess import remove_noise_blocks, strip_front_matter


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
