from tradutor.section_splitter import split_into_sections


def test_split_into_sections_detects_titles() -> None:
    text = "Front\n\nPrologue\nLine A\n\nChapter 1: Start\nLine B\n\nEpilogue\nLine C"
    sections = split_into_sections(text)
    assert len(sections) == 3
    assert sections[0]["title"].lower().startswith("prologue")
    assert "Line A" in sections[0]["body"]
    assert sections[1]["title"].lower().startswith("chapter 1")
    assert "Line B" in sections[1]["body"]
    assert sections[2]["title"].lower().startswith("epilogue")
    assert "Line C" in sections[2]["body"]


def test_split_into_sections_when_no_markers_returns_full_text() -> None:
    text = "No markers here.\nJust text."
    sections = split_into_sections(text)
    assert len(sections) == 1
    assert sections[0]["title"] == "Full Text"
    assert "No markers" in sections[0]["body"]
