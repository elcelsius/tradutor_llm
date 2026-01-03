import json
import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.debug_run import DebugRunWriter
from tradutor.llm_backend import LLMResponse
from tradutor.refine import refine_markdown_file
from tradutor.translate import translate_document
from tradutor.utils import setup_logging, write_text, read_text


class FakeTranslateBackend:
    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text="### TEXTO_TRADUZIDO_INICIO\nTexto traduzido em português.\n\nSegundo parágrafo traduzido.\n### TEXTO_TRADUZIDO_FIM",
            latency=0.01,
        )


class FakeRefineBackend:
    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text="### TEXTO_REFINADO_INICIO\nTexto refinado.\n### TEXTO_REFINADO_FIM",
            latency=0.01,
        )


def test_debug_run_translate_refine_outputs(tmp_path: Path) -> None:
    cfg = AppConfig(output_dir=tmp_path)
    logger = setup_logging(logging.DEBUG)
    source_text = "First paragraph in English.\n\nSecond paragraph in English."

    result_plain = translate_document(
        pdf_text=source_text,
        backend=FakeTranslateBackend(),
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        already_preprocessed=True,
    )

    debug_run = DebugRunWriter.create(
        output_dir=tmp_path,
        slug="sample",
        input_kind="md",
        max_chunks=None,
        max_chars_per_file=5000,
        store_llm_raw=True,
    )
    debug_run.preprocessed_rel = "10_preprocess/sample_preprocessed.md"
    debug_run.desquebrado_rel = "20_desquebrar/sample_raw_desquebrado.md"
    debug_run.write_text(debug_run.preprocessed_rel, source_text)
    debug_run.write_text(debug_run.desquebrado_rel, source_text)

    result_debug = translate_document(
        pdf_text=source_text,
        backend=FakeTranslateBackend(),
        cfg=cfg,
        logger=logger,
        source_slug="sample",
        already_preprocessed=True,
        debug_run=debug_run,
    )

    assert result_plain == result_debug

    input_path = tmp_path / "sample_pt.md"
    output_path = tmp_path / "sample_pt_refinado.md"
    write_text(input_path, result_debug)
    debug_run.pt_output_rel = "sample_pt.md"

    refine_markdown_file(
        input_path=input_path,
        output_path=output_path,
        backend=FakeRefineBackend(),
        cfg=cfg,
        logger=logger,
        cleanup_mode="off",
        debug_run=debug_run,
    )

    translate_manifest_path = debug_run.run_dir / "40_translate" / "translate_manifest.json"
    refine_manifest_path = debug_run.run_dir / "60_refine" / "refine_manifest.json"
    assert translate_manifest_path.exists()
    assert refine_manifest_path.exists()

    translate_manifest = json.loads(read_text(translate_manifest_path))
    refine_manifest = json.loads(read_text(refine_manifest_path))
    assert translate_manifest["chunking"]["total_chunks"] == len(translate_manifest["chunks"])
    assert refine_manifest["refine"]["total_chunks"] == len(refine_manifest["chunks"])

    debug_chunk = debug_run.run_dir / "40_translate" / "debug_traducao" / "chunk001_final_pt.txt"
    assert debug_chunk.exists()

    for path_value in translate_manifest["input_paths"].values():
        assert path_value is None or not Path(path_value).is_absolute()
    for chunk in translate_manifest["chunks"]:
        for key in ("debug_original", "debug_context", "debug_llm_raw", "debug_final"):
            assert not Path(chunk["outputs"][key]).is_absolute()
    for key, value in refine_manifest["input_paths"].items():
        if value is not None:
            assert not Path(value).is_absolute()
