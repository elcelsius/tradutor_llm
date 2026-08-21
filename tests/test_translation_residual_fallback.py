import logging
import re

import tradutor.translate as translate
from tradutor.config import AppConfig


class _StubBackend:
    """Processamento interno auxiliar."""

    backend = "stub"
    model = "stub"
    num_predict = 128
    temperature = 0.1
    repeat_penalty = 1.0


def _extract_block(prompt: str) -> str:
    """Processamento interno auxiliar."""
    match = re.search(r'TEXTO A SER TRADUZIDO:\n"""(.*?)"""', prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def test_targeted_residual_english_fallback_replaces_only_leaked_dialogue(
    monkeypatch, tmp_path
) -> None:
    """Processamento interno auxiliar."""
    backend = _StubBackend()
    labels: list[str] = []
    prompts: list[str] = []

    def fake_call_with_retry(backend, prompt, cfg, logger, label):
        """Processamento interno auxiliar."""
        labels.append(label)
        prompts.append(prompt)
        source = _extract_block(prompt)
        if label.startswith("trad-residual-"):
            output = (
                "\u201cNyaki\u2026 Nyaki\u2014me desculpe\u2026 me desculpe por ter demorado tanto para encontrar voce\u2026! "
                "Me desculpe\u2026 sinto muito!\u201d"
            )
        else:
            output = source
        raw = f"### TEXTO_TRADUZIDO_INICIO\n{output}\n### TEXTO_TRADUZIDO_FIM"
        return raw, output, 1, None

    monkeypatch.setattr(translate, "_call_with_retry", fake_call_with_retry)
    cfg = AppConfig(
        output_dir=tmp_path,
        split_by_sections=False,
        max_retries=1,
        translate_chunk_chars=5000,
        use_translation_repair=False,
        fail_on_chunk_error=True,
    )
    source = (
        "Ela correu para a tenda.\n\n"
        "\u201cNyaki\u2026 Nyaki\u2014I'm sorry\u2026 I'm sorry it took me so long to find you\u2026! "
        "I'm sorry\u2026 I'm so sorry!\u201d"
    )

    result = translate.translate_document(
        pdf_text=source,
        backend=backend,
        cfg=cfg,
        logger=logging.getLogger("residual-fallback"),
        source_slug="residual-fallback",
        already_preprocessed=True,
    )

    assert "I'm sorry" not in result
    assert "me desculpe" in result
    assert any(label.startswith("trad-residual-") for label in labels)
    residual_prompt = next(
        prompt for label, prompt in zip(labels, prompts) if label.startswith("trad-residual-")
    )
    assert "TRECHO RESIDUAL EM INGLÊS" in residual_prompt
    assert "I'm sorry" in residual_prompt
