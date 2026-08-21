from tradutor.section_splitter import split_into_sections


def test_split_into_sections_detects_titles() -> None:
    """Processamento interno auxiliar."""
    text = "Front\n\nPrologue\nLine A\n\nChapter 1: Start\nLine B\n\nEpilogue\nLine C"
    sections = split_into_sections(text)
    assert len(sections) == 4  # inclui preâmbulo
    assert sections[0]["title"] == "Full Text"
    assert sections[1]["title"].lower().startswith("prologue")
    assert "Line A" in sections[1]["body"]
    assert sections[2]["title"].lower().startswith("chapter 1")
    assert "Line B" in sections[2]["body"]
    assert sections[3]["title"].lower().startswith("epilogue")
    assert "Line C" in sections[3]["body"]


def test_split_into_sections_when_no_markers_returns_full_text() -> None:
    """Processamento interno auxiliar."""
    text = "No markers here.\nJust text."
    sections = split_into_sections(text)
    assert len(sections) == 1
    assert sections[0]["title"] == "Full Text"
    assert "No markers" in sections[0]["body"]


def test_split_adds_preamble_before_first_marker() -> None:
    """Processamento interno auxiliar."""
    text = "Intro text here.\n\nChapter 1:\nBody of chapter one."
    sections = split_into_sections(text)
    assert len(sections) == 2
    assert sections[0]["title"] == "Full Text"
    assert "Intro text" in sections[0]["body"]
    assert sections[1]["title"].lower().startswith("chapter 1")
    assert "Body of chapter one" in sections[1]["body"]


def test_split_handles_colon_without_title_and_epilogue() -> None:
    """Processamento interno auxiliar."""
    text = "Chapter 1:\nBody\n\nEpilogue\nThe end."
    sections = split_into_sections(text)
    assert len(sections) == 2
    assert sections[0]["title"].lower().startswith("chapter 1")
    assert sections[1]["title"].lower().startswith("epilogue")


def test_split_ignores_empty_toc_entries() -> None:
    """Processamento interno auxiliar."""
    text = "Prologue\n\nChapter 1\n\nEpilogue\nClosing text."
    sections = split_into_sections(text)
    # Prologue e Chapter 1 sem corpo são ignorados; mantém Epilogue
    assert len(sections) == 1
    assert sections[0]["title"].lower().startswith("epilogue")
    assert "Closing text." in sections[0]["body"]


def test_split_ignores_numeric_toc_entries() -> None:
    """Processamento interno auxiliar."""
    text = "Chapter 1\n1\nChapter 2\n2\nChapter 3\nBody text."
    sections = split_into_sections(text)
    assert len(sections) == 1
    assert sections[0]["title"].lower().startswith("chapter 3")
    assert "Body text." in sections[0]["body"]


def test_split_marks_chapter_subtitles_separately_from_narration() -> None:
    text = (
        "Chapter 2:\n\nSwirling Wills I\n\nDESCENDED FROM the carriage.\n\n"
        "Chapter 5:\n\nConnections\n\nMY MIND resurfaced."
    )

    sections = split_into_sections(text)

    assert sections[0]["title"] == "Chapter 2:"
    assert sections[0]["body"].startswith(
        "# Chapter 2:\n\n## Swirling Wills\n\nI descended from"
    )
    assert sections[1]["title"] == "Chapter 5:"
    assert sections[1]["body"].startswith(
        "# Chapter 5:\n\n## Connections\n\nMy mind resurfaced."
    )


def test_split_preserves_small_caps_entity_at_section_opening() -> None:
    sections = split_into_sections("Chapter 1:\n\nMIMORI TOUKA had departed.")

    assert "MIMORI TOUKA had departed." in sections[0]["body"]


def test_split_separates_inline_pov_from_small_caps_opening() -> None:
    text = (
        "Chapter 1:\n"
        "Mimori Touka THE NEXT MORNING… The army was ready to depart.\n\n"
        "“Thanks, Sogou.” The Goddess Vicius IT WAS ON the morning that preparations ended."
    )

    sections = split_into_sections(text)

    assert "## Mimori Touka\n\nThe next morning… The army was ready" in sections[0]["body"]
    assert "## The Goddess Vicius\n\nIt was on the morning" in sections[0]["body"]
    assert "“Thanks, Sogou.”\n\n## The Goddess Vicius" in sections[0]["body"]
