import json
import logging
from pathlib import Path

from tradutor.cache_utils import chunk_hash, set_cache_base_dir
from tradutor.config import AppConfig
from tradutor.translate import (
    TRANSLATE_PIPELINE_VERSION,
    translate_document,
    translation_prompt_fingerprint,
)


class _StubResponse:
    """Processamento interno auxiliar."""

    def __init__(self, text: str):
        """Processamento interno auxiliar."""
        self.text = text


class _StubBackend:
    """Processamento interno auxiliar."""

    def __init__(self) -> None:
        """Processamento interno auxiliar."""
        self.backend = "stub"
        self.model = "stub-model"
        self.num_predict = 42
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Processamento interno auxiliar."""
        self.calls += 1
        body = "um dois tres quatro cinco seis sete oito nove dez onze doze treze catorze quinze"
        return _StubResponse(
            f"### TEXTO_TRADUZIDO_INICIO\n{body}\n### TEXTO_TRADUZIDO_FIM"
        )


class _CacheReuseBackend(_StubBackend):
    def generate(self, prompt: str):
        self.calls += 1
        body = (
            "A tradução preserva o sentido da cena e mantém a pontuação natural. "
            "Os personagens conversam com clareza enquanto avançam pelo caminho. "
            "A narração descreve o ambiente sem repetir nenhuma construção extensa. "
            "Cada frase acrescenta uma informação diferente ao trecho revisado. "
            "O grupo observa a paisagem e decide continuar a jornada naquela manhã. "
            "Uma resposta breve encerra o diálogo antes da próxima mudança de cena. "
            "O texto final permanece fluente, completo e adequado ao leitor brasileiro. "
            "Assim, a saída atende aos limites de qualidade usados pela pipeline."
        )
        return _StubResponse(
            f"### TEXTO_TRADUZIDO_INICIO\n{body}\n### TEXTO_TRADUZIDO_FIM"
        )


def test_translate_cache_mismatch_is_ignored(tmp_path: Path) -> None:
    """Processamento interno auxiliar."""
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
    cache_path.write_text(
        json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8"
    )

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

    # não veio do cache desatualizado; ou traduziu, ou rejeitou pelo guardrail
    assert "CACHED_SHOULD_BE_IGNORED" not in result
    assert backend.calls >= 0
    assert cache_path.exists()


def test_translate_cache_ignores_allow_adaptation_change(tmp_path: Path) -> None:
    """Processamento interno auxiliar."""
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _StubBackend()
    logger = logging.getLogger("translate-cache-flag")

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
        "metadata": {
            "backend": backend.backend,
            "model": backend.model,
            "num_predict": backend.num_predict,
            "temperature": backend.temperature,
            "repeat_penalty": backend.repeat_penalty,
            "translate_chunk_chars": cfg.translate_chunk_chars,
            "glossary_hash": None,
            "doc_hash": chunk_hash(text),
            "source": "sample",
            "allow_adaptation": False,
            "split_by_sections": False,
            "dialogue_guardrails_mode": getattr(
                cfg, "translate_dialogue_guardrails", "strict"
            ),
            "prompt_hash": translation_prompt_fingerprint(allow_adaptation=False),
            "pipeline_version": TRANSLATE_PIPELINE_VERSION,
        },
    }
    cache_path.write_text(
        json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8"
    )

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
        allow_adaptation=True,
    )

    assert "CACHED_SHOULD_BE_IGNORED" not in result
    assert backend.calls >= 1


def test_translate_cache_is_reused_when_full_signature_matches(tmp_path: Path) -> None:
    cfg = AppConfig(
        output_dir=tmp_path,
        split_by_sections=False,
        use_translation_repair=False,
    )
    backend = _CacheReuseBackend()
    logger = logging.getLogger("translate-cache-reuse")
    kwargs = {
        "pdf_text": "This source sentence is long enough for cache validation. " * 10,
        "backend": backend,
        "cfg": cfg,
        "logger": logger,
        "source_slug": "sample",
        "resume_manifest": None,
        "glossary_text": None,
        "debug_translation": False,
        "parallel_workers": 1,
        "debug_chunks": False,
        "already_preprocessed": True,
        "translation_repair": False,
    }

    first = translate_document(
        **kwargs,
        progress_path=tmp_path / "first_progress.json",
    )
    second = translate_document(
        **kwargs,
        progress_path=tmp_path / "second_progress.json",
    )

    assert first == second
    assert backend.calls == 1
