from tradutor.glossary_utils import select_terms_for_chunk


def test_glossary_word_boundary_matching() -> None:
    manual_terms = [{"key": "Art", "pt": "Arte"}]
    chunk_text = "This is a partial mention that should not match."
    selected, matched = select_terms_for_chunk(manual_terms, chunk_text, match_limit=80, fallback_limit=0)
    assert matched == 0
    assert selected == []
