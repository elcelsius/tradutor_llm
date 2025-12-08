import textwrap

from tradutor.preprocess import preprocess_text


def test_preprocess_removes_watermarks_but_keeps_text() -> None:
    raw = textwrap.dedent(
        """
        They say nothing good can ever come from revenge.
        Page 190
        Goldenagato | mp4directs.com
        Foul Goddess…
        I will have my revenge.
        Page 191
        Goldenagato | mp4directs.com
        """
    ).strip()

    cleaned = preprocess_text(raw, logger=None)

    assert "Page 190" not in cleaned
    assert "Goldenagato" not in cleaned
    assert "mp4directs" not in cleaned

    assert "revenge" in cleaned
    assert "Foul Goddess" in cleaned
    assert "I will have my revenge" in cleaned
