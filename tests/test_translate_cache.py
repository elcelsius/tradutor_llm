import json
import logging
from pathlib import Path

from tradutor.cache_utils import chunk_hash, set_cache_base_dir
from tradutor.config import AppConfig
from tradutor.translate import translate_document


class _StubResponse:
    def __init__(self, text: str):
        self.text = text


class _StubBackend:
    def __init__(self) -> None:
        self.backend = "stub"
        self.model = "stub-model"
        self.num_predict = 42
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        self.calls += 1
        body = "um dois tres quatro cinco seis sete oito nove dez onze doze treze catorze quinze"
        return _StubResponse(f"### TEXTO_TRADUZIDO_INICIO\n{body}\n### TEXTO_TRADUZIDO_FIM")


def test_translate_cache_mismatch_is_ignored(tmp_path: Path) -> None:
    cfg = AppConfig(output_dir=tmp_path)
    backend = _StubBackend()
    logger = logging.getLogger("translate-cache")

    text = "Hello world."
    h = chunk_hash(text)
    set_cache_base_dir(tmp_path)
    cache_path = tmp_path / "cache_traducao" / f"{h}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "hash": h,
        "raw_output": "RAW",
        "final_output": "CACHED_SHOULD_BE_IGNORED",
        "timestamp": "now",
        "metadata": {"backend": "other-backend"},
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8")

    result = translate_document(
        pdf_text=text,
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

    assert "um dois tres" in result  # veio do backend, nao do cache desatualizado
    assert backend.calls == 1
    assert cache_path.exists()
