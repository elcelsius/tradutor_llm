import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.translate import translate_document


class _PromptCaptureBackend:
    def __init__(self) -> None:
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 128
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.prompts: list[str] = []

    def generate(self, prompt: str):
        self.prompts.append(prompt)
        return type("Resp", (), {"text": "### TEXTO_TRADUZIDO_INICIO\nTradução do escudo.\n### TEXTO_TRADUZIDO_FIM"})


def test_glossary_injects_only_matching_terms(tmp_path: Path) -> None:
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _PromptCaptureBackend()
    logger = logging.getLogger("glossary-context")
    manual_terms = [
        {"key": "Shield", "pt": "Escudo"},
        {"key": "Sword", "pt": "Espada"},
    ]
    input_text = ("The shield was heavy and sturdy. " * 10).strip()

    translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        glossary_manual_terms=manual_terms,
    )

    assert backend.prompts, "espera ao menos um prompt enviado ao backend"
    prompt = backend.prompts[0]
    assert "Shield" in prompt and "Escudo" in prompt
    assert "Sword" not in prompt


def test_glossary_matches_aliases(tmp_path: Path) -> None:
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _PromptCaptureBackend()
    logger = logging.getLogger("glossary-alias")
    manual_terms = [
        {"key": "Magic Sword", "pt": "Espada Mágica", "aliases": ["Blade of Dawn", "Dawnblade"]},
        {"key": "Shield", "pt": "Escudo"},
    ]
    input_text = ("The Blade of Dawn was legendary and revered across the lands. " * 8).strip()

    translate_document(
        pdf_text=input_text,
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        glossary_manual_terms=manual_terms,
    )

    assert backend.prompts, "espera ao menos um prompt enviado ao backend"
    prompt = backend.prompts[0]
    assert "Magic Sword" in prompt and "Espada Mágica" in prompt
    assert "Shield" not in prompt
