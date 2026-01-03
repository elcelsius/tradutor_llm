import sys
import types

import pytest

from tradutor.config import AppConfig
from tradutor.utils import setup_logging


def _install_reportlab_stub() -> None:
    fake_reportlab = types.ModuleType("reportlab")
    fake_lib = types.ModuleType("reportlab.lib")
    fake_enums = types.ModuleType("reportlab.lib.enums")
    fake_enums.TA_JUSTIFY = 0
    fake_enums.TA_LEFT = 0
    fake_pagesizes = types.ModuleType("reportlab.lib.pagesizes")
    fake_pagesizes.A4 = (0, 0)
    fake_styles = types.ModuleType("reportlab.lib.styles")

    class _DummyStyleSheet:
        def add(self, *args, **kwargs):
            return None

    class _DummyParagraphStyle:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name", "dummy")

    fake_styles.ParagraphStyle = _DummyParagraphStyle
    fake_styles.StyleSheet1 = _DummyStyleSheet
    fake_styles.getSampleStyleSheet = lambda: {}
    fake_units = types.ModuleType("reportlab.lib.units")
    fake_units.cm = 1
    fake_platypus = types.ModuleType("reportlab.platypus")
    fake_platypus.Paragraph = lambda *args, **kwargs: None
    fake_platypus.SimpleDocTemplate = lambda *args, **kwargs: None
    fake_platypus.Spacer = lambda *args, **kwargs: None
    fake_pdfbase = types.ModuleType("reportlab.pdfbase")
    fake_pdfmetrics = types.ModuleType("reportlab.pdfbase.pdfmetrics")
    fake_pdfmetrics.registerFont = lambda *args, **kwargs: None
    fake_ttfonts = types.ModuleType("reportlab.pdfbase.ttfonts")
    fake_ttfonts.TTFont = lambda *args, **kwargs: None
    fake_pdfbase.pdfmetrics = fake_pdfmetrics
    fake_pdfbase.ttfonts = fake_ttfonts

    fake_reportlab.lib = fake_lib
    fake_reportlab.pdfbase = fake_pdfbase
    fake_lib.enums = fake_enums
    fake_lib.pagesizes = fake_pagesizes
    fake_lib.styles = fake_styles
    fake_lib.units = fake_units

    sys.modules.update(
        {
            "reportlab": fake_reportlab,
            "reportlab.lib": fake_lib,
            "reportlab.lib.enums": fake_enums,
            "reportlab.lib.pagesizes": fake_pagesizes,
            "reportlab.lib.styles": fake_styles,
            "reportlab.lib.units": fake_units,
            "reportlab.platypus": fake_platypus,
            "reportlab.pdfbase": fake_pdfbase,
            "reportlab.pdfbase.pdfmetrics": fake_pdfmetrics,
            "reportlab.pdfbase.ttfonts": fake_ttfonts,
        }
    )


def test_run_translate_uses_desquebrar_before_translate(monkeypatch, tmp_path):
    _install_reportlab_stub()
    import tradutor.main as main  # noqa: WPS433 (import inside test for stub)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("dummy", encoding="utf-8")

    cfg = AppConfig(data_dir=tmp_path, output_dir=tmp_path)
    logger = setup_logging()

    calls: dict[str, object] = {}

    def fake_extract_pdf_text(path, logger):
        return "raw pdf text"

    def fake_preprocess_text(text, logger=None, **kwargs):
        if kwargs.get("return_stats"):
            return "preprocessed text", {"chars_in": len(text), "chars_out": len("preprocessed text")}
        return "preprocessed text"

    def fake_desquebrar_text(text, cfg, logger, backend, chunk_chars=None):
        calls["chunk_chars"] = chunk_chars
        return "texto desquebrado", types.SimpleNamespace(total_chunks=1, cache_hits=0, fallbacks=0)

    def fake_translate_document(pdf_text, backend, cfg, logger, **kwargs):
        calls["translated_input"] = pdf_text
        calls["already_preprocessed"] = kwargs.get("already_preprocessed")
        return "conteudo traduzido"

    class DummyBackend:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt):
            pytest.fail("LLMBackend.generate should not be called in this test")

    monkeypatch.setattr(main, "extract_pdf_text", fake_extract_pdf_text)
    monkeypatch.setattr(main, "preprocess_text", fake_preprocess_text)
    monkeypatch.setattr(main, "desquebrar_text", fake_desquebrar_text)
    monkeypatch.setattr(main, "translate_document", fake_translate_document)
    monkeypatch.setattr(main, "LLMBackend", DummyBackend)

    args = types.SimpleNamespace(
        command="traduz",
        input=None,
        backend="ollama",
        model="model-x",
        num_predict=128,
        no_refine=True,
        resume=False,
        use_glossary=False,
        manual_glossary=None,
        parallel=1,
        preprocess_advanced=False,
        cleanup_before_refine=None,
        debug_chunks=False,
        debug=False,
        request_timeout=30,
        use_desquebrar=True,
        desquebrar_backend="ollama",
        desquebrar_model="modelo-desq",
        desquebrar_temperature=0.1,
        desquebrar_chunk_chars=777,
        desquebrar_num_predict=256,
        desquebrar_repeat_penalty=1.1,
    )

    main.run_translate(args, cfg, logger)

    assert calls["translated_input"] == "texto desquebrado"
    assert calls["already_preprocessed"] is True
    assert calls["chunk_chars"] == 777


def test_run_translate_skips_desquebrar_when_disabled(monkeypatch, tmp_path):
    _install_reportlab_stub()
    import tradutor.main as main  # noqa: WPS433

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("dummy", encoding="utf-8")

    cfg = AppConfig(data_dir=tmp_path, output_dir=tmp_path)
    logger = setup_logging()

    calls: dict[str, object] = {}

    def fake_extract_pdf_text(path, logger):
        return "raw pdf text"

    def fake_preprocess_text(text, logger=None, **kwargs):
        if kwargs.get("return_stats"):
            return "preprocessed text", {"chars_in": len(text), "chars_out": len("preprocessed text")}
        return "preprocessed text"

    def fake_translate_document(pdf_text, backend, cfg, logger, **kwargs):
        calls["translated_input"] = pdf_text
        calls["already_preprocessed"] = kwargs.get("already_preprocessed")
        return "conteudo traduzido"

    class DummyBackend:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt):
            pytest.fail("LLMBackend.generate should not be called in this test")

    monkeypatch.setattr(main, "extract_pdf_text", fake_extract_pdf_text)
    monkeypatch.setattr(main, "preprocess_text", fake_preprocess_text)
    monkeypatch.setattr(main, "desquebrar_text", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not call")))
    monkeypatch.setattr(main, "translate_document", fake_translate_document)
    monkeypatch.setattr(main, "LLMBackend", DummyBackend)

    args = types.SimpleNamespace(
        command="traduz",
        input=None,
        backend="ollama",
        model="model-x",
        num_predict=128,
        no_refine=True,
        resume=False,
        use_glossary=False,
        manual_glossary=None,
        parallel=1,
        preprocess_advanced=False,
        cleanup_before_refine=None,
        debug_chunks=False,
        debug=False,
        request_timeout=30,
        use_desquebrar=False,
        desquebrar_backend="ollama",
        desquebrar_model="modelo-desq",
        desquebrar_temperature=0.1,
        desquebrar_chunk_chars=777,
        desquebrar_num_predict=256,
        desquebrar_repeat_penalty=1.1,
    )

    main.run_translate(args, cfg, logger)

    assert calls["translated_input"] == "preprocessed text"
    assert calls["already_preprocessed"] is True
