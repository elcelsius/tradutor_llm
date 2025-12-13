from tradutor.pdf import normalize_markdown_for_pdf, _inline_markdown_to_html


def test_normalize_markdown_for_pdf_converts_br_and_paragraphs():
    text = "Linha 1<br/>\nLinha 2\n\nLinha 3<br>Continua"
    parts = normalize_markdown_for_pdf(text)
    assert parts == ["Linha 1", "Linha 2", "Linha 3\nContinua"]


def test_inline_markdown_to_html_preserves_simple_tags():
    src = "**negrito** e _italico_"
    html = _inline_markdown_to_html(src)
    assert "<b>negrito</b>" in html
    assert "<i>italico</i>" in html
