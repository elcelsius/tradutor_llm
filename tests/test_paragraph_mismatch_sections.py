import logging

from tradutor.config import AppConfig
from tradutor.translate import translate_document
from tradutor.utils import setup_logging


class _StubBackend:
    """Processamento interno auxiliar."""

    def __init__(self) -> None:
        """Processamento interno auxiliar."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 256
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Processamento interno auxiliar."""
        # Sempre retorna um único parágrafo, independentemente do chunk original
        self.calls += 1
        return type(
            "Resp",
            (),
            {
                "text": "### TEXTO_TRADUZIDO_INICIO\nPARAGRAFO UNICO\n### TEXTO_TRADUZIDO_FIM",
            },
        )


def test_paragraph_mismatch_detects_multi_section_loss(caplog, tmp_path) -> None:
    """Processamento interno auxiliar."""
    logger = setup_logging(logging.ERROR)
    cfg = AppConfig(
        split_by_sections=True, translate_chunk_chars=5000, output_dir=tmp_path
    )
    text = "Prologue\nA\n\nB\n\nC\n\nChapter 1\nD"

    backend = _StubBackend()
    with caplog.at_level(logging.ERROR):
        translate_document(
            pdf_text=text,
            backend=backend,
            cfg=cfg,
            logger=logger,
            already_preprocessed=True,
        )

    assert any(
        "Paragrafos ausentes apos traducao" in rec.message for rec in caplog.records
    )
    # Há pelo menos uma chamada por seção; retries podem ocorrer para a perda
    # estrutural que este teste induz deliberadamente.
    assert backend.calls >= 2
