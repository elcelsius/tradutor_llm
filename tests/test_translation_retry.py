import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.translate import translate_document


class _RetryBackend:
    def __init__(self) -> None:
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        self.calls += 1
        if self.calls == 1:
            # output truncado forçando retry (ratio baixo)
            text = "### TEXTO_TRADUZIDO_INICIO\nOi.\n### TEXTO_TRADUZIDO_FIM"
        else:
            text = "### TEXTO_TRADUZIDO_INICIO\nTexto completo traduzido.\n### TEXTO_TRADUZIDO_FIM"
        return type("Resp", (), {"text": text})


class _MissingMarkerBackend:
    def __init__(self) -> None:
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 10
        self.temperature = 0.1
        self.repeat_penalty = 1.0

    def generate(self, prompt: str):
        text = "### TEXTO_TRADUZIDO_INICIO\nTitulo traduzido sem marcador final"
        return type("Resp", (), {"text": text})


def test_translate_retries_on_truncated_output(tmp_path: Path) -> None:
    cfg = AppConfig(output_dir=tmp_path, max_retries=2, split_by_sections=False)
    backend = _RetryBackend()
    logger = logging.getLogger("retry-test")
    input_text = "This is a longer input text that should be fully present after translation."

    result = translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        progress_path=None,
        resume_manifest=None,
        glossary_text=None,
        debug_translation=False,
        parallel_workers=1,
        debug_chunks=False,
        already_preprocessed=True,
    )

    assert "Texto completo traduzido" in result
    assert backend.calls >= 2


def test_translate_parses_output_without_end_marker(tmp_path: Path) -> None:
    cfg = AppConfig(output_dir=tmp_path, max_retries=1, split_by_sections=False)
    backend = _MissingMarkerBackend()
    logger = logging.getLogger("missing-marker")
    input_text = "Epilogue"

    result = translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        progress_path=None,
        resume_manifest=None,
        glossary_text=None,
        debug_translation=False,
        parallel_workers=1,
        debug_chunks=False,
        already_preprocessed=True,
    )

    assert "Titulo traduzido" in result
    assert "TEXTO_TRADUZIDO" not in result
