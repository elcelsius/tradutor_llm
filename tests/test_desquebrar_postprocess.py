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


def _run_desquebrar(original, llm_output):
    cfg = AppConfig(desquebrar_chunk_chars=500)
    logger = setup_logging()
    backend = _StubBackend([llm_output])
    result, stats = desquebrar_text(original, cfg, logger, backend)
    return result, stats, backend


def test_desquebrar_stray_quote_line_vol8_case():
    original = "“devices.”\n“Oh, what…”"
    llm_output = "“devices.”\n\"\n“Oh, what…”."

    result, stats, backend = _run_desquebrar(original, llm_output)

    assert backend.calls == 1
    assert stats.fallbacks == 1
    assert stats.blocks[0]["fallback_reason"] == "qa_stray_quote_lines"
    assert result == "“devices.”\n\n“Oh, what…”"


def test_desquebrar_fixes_stutter_space():
    original = "D- do."
    llm_output = "D- do."

    result, stats, _ = _run_desquebrar(original, llm_output)

    assert stats.fallbacks == 0
    assert result == "D-do."


def test_desquebrar_fixes_hyphen_linewrap():
    original = "hang-\nups"
    llm_output = "hang-\nups"

    result, stats, _ = _run_desquebrar(original, llm_output)

    assert stats.fallbacks == 0
    assert result == "hang-ups"


def test_desquebrar_isolates_asterisks():
    original = "alpha\n***\nomega"
    llm_output = "alpha\n***\nomega"

    result, stats, _ = _run_desquebrar(original, llm_output)

    assert stats.fallbacks == 0
    assert result == "alpha\n\n***\n\nomega"
