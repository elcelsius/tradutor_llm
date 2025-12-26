import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.translate import translate_document


class _ParryBackend:
    def __init__(self) -> None:
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 128
        self.temperature = 0.1
        self.repeat_penalty = 1.0

    def generate(self, prompt: str):
        return type("Resp", (), {"text": "### TEXTO_TRADUZIDO_INICIO\nEle parriu o golpe rapidamente.\n### TEXTO_TRADUZIDO_FIM"})


def test_postprocess_fixes_parry_false_cognate(tmp_path: Path) -> None:
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _ParryBackend()
    logger = logging.getLogger("parry-fix")
    input_text = "He parried the incoming blow."

    result = translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
    )

    assert "aparou" in result
    assert "parriu" not in result
