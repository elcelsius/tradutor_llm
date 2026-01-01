import types

from tradutor.config import AppConfig
from tradutor.desquebrar import desquebrar_text
from tradutor.desquebrar_safe import safe_reflow
from tradutor.utils import setup_logging


class _StubBackend:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def generate(self, prompt):
        out = self.outputs[self.calls]
        self.calls += 1
        return types.SimpleNamespace(text=out)


def test_desquebrar_fallback_on_stray_quote_line():
    cfg = AppConfig(desquebrar_chunk_chars=500)
    logger = setup_logging()
    original = "Mechanical devices.\nOh, what a savage prospect…"
    bad_output = 'Mechanical devices.\n"\nOh, what a savage prospect…'
    backend = _StubBackend([bad_output])

    result, stats = desquebrar_text(original, cfg, logger, backend)

    assert result.replace("\n\n", "\n") == safe_reflow(original)
    assert stats.fallbacks == 1
    assert stats.stray_quote_lines >= 1


def test_desquebrar_hyphen_and_asterisks_postprocess(tmp_path):
    cfg = AppConfig(desquebrar_chunk_chars=500, output_dir=tmp_path)
    logger = setup_logging()
    original = "D- do\nhang-\nups\nbefore\n***\nafter"
    backend = _StubBackend([original])

    result, stats = desquebrar_text(original, cfg, logger, backend)

    assert "D-do" in result
    assert "hang-ups" in result
    assert "\nbefore\n\n***\n\nafter" in result
    assert stats.fallbacks == 0
    assert stats.hyphen_linewrap_count >= 1
