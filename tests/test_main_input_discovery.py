from pathlib import Path

from tradutor.main import find_pdfs


def test_find_pdfs_accepts_a_directory_of_pdfs(tmp_path: Path) -> None:
    inputs = tmp_path / "recortes"
    inputs.mkdir()
    (inputs / "b.pdf").write_bytes(b"%PDF-test")
    (inputs / "a.pdf").write_bytes(b"%PDF-test")
    (inputs / "ignore.txt").write_text("ignore", encoding="utf-8")

    result = find_pdfs(tmp_path, str(inputs))

    assert [path.name for path in result] == ["a.pdf", "b.pdf"]


def test_find_pdfs_keeps_a_specific_pdf_path(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-test")

    assert find_pdfs(tmp_path, str(pdf)) == [pdf]
