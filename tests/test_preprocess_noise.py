import types
from pathlib import Path

import pytest

from tradutor.config import AppConfig
from tradutor.llm_backend import LLMResponse
from tradutor.preprocess import preprocess_text
from tradutor.utils import setup_logging, read_text


def test_preprocess_removes_watermarks_and_toc() -> None:
    raw = "\n".join(
        [
            "OceanofPDF.com",
            "Some normal paragraph from the book.",
            "Table of Contents",
            "Chapter 1",
            "Chapter 2",
            "",
            "Sign up for our newsletter",
            "Another real paragraph.",
            "gomanga.com/",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert "OceanofPDF" not in cleaned
    assert "Table of Contents" not in cleaned
    assert "Chapter 1" not in cleaned
    assert "Sign up for" not in cleaned
    assert "gomanga.com" not in cleaned
    assert "Some normal paragraph" in cleaned
    assert "Another real paragraph." in cleaned
    assert stats["oceanofpdf_removed_count"] >= 1
    assert stats["toc_blocks_removed_count"] >= 1


def test_preprocess_removes_oceanofpdf_variations() -> None:
    raw = "\n".join(
        [
            " OCEANOF PDF . COM ",
            "\xa0OceanofPDF.com\xa0",
            "Story continues.",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert "OceanofPDF" not in cleaned
    assert stats["oceanofpdf_removed_count"] >= 2


def test_preprocess_removes_newsletter_and_gomanga_url() -> None:
    raw = "\n".join(
        [
            "Thank you for reading!",
            "Sign up for updates at https://gomanga.com/newsletter",
            "Another line with http://discord.gg/something",
            "Real story stays.",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert "gomanga" not in cleaned.lower()
    assert "discord.gg" not in cleaned.lower()
    assert "thank you for reading" not in cleaned.lower()
    assert "Real story stays." in cleaned
    assert stats["promo_lines_removed_count"] >= 2


def test_preprocess_removes_tail_toc_block() -> None:
    tail_toc = "\n".join(
        [
            "Table of Contents",
            "Color Inserts",
            "Title Page",
            "Prologue 1",
            "Chapter 1",
            "Chapter 2",
            "Chapter 3",
            "Chapter 4",
            "Chapter 5",
            "Chapter 6",
            "Afterword",
            "Newsletter",
        ]
    )
    raw = "\n".join(
        [
            "Real narrative starts.",
            "Keeps going with content.",
            tail_toc,
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert "Table of Contents" not in cleaned
    assert "Chapter 1" not in cleaned
    assert "Afterword" not in cleaned
    assert "Real narrative starts." in cleaned
    assert stats["toc_blocks_removed_count"] >= 1


def test_preprocess_removes_tail_newsletter_block() -> None:
    raw = "\n".join(
        [
            "Story core stays before.",
            "Get the latest news about your favorite Seven Seas books and brand-new",
            "licenses delivered to your inbox every week:",
            "Or visit us online:",
            "Story core stays after.",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert "Get the latest news" not in cleaned
    assert "licenses delivered to your inbox every week" not in cleaned
    assert "visit us online" not in cleaned.lower()
    assert "Story core stays before." in cleaned
    assert "Story core stays after." in cleaned
    assert stats["promo_lines_removed_count"] >= 3


def test_preprocess_keeps_narrative_with_contents_word() -> None:
    raw = "\n".join(
        [
            "The book's contents were mysterious and deep.",
            "Nothing promotional here.",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "contents were mysterious" in cleaned


def test_preprocess_removes_repeated_headers_and_keeps_story() -> None:
    raw = "\n".join(
        [
            *["SCAN GROUP" for _ in range(6)],
            "A real story line that should stay.",
            "Another normal paragraph follows.",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert "SCAN GROUP" not in cleaned
    assert "real story line" in cleaned
    assert stats["repeated_lines_removed_count"] >= 6
    assert stats["top_repeated_lines"]


def test_preprocess_preserves_dialogue_with_sign_up_phrase() -> None:
    raw = "\n".join(
        [
            '"Sign up for glory," she whispered.',
            "Sign up for our newsletter",
            "Nothing else changes.",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert '"Sign up for glory," she whispered.' in cleaned
    assert "Sign up for our newsletter" not in cleaned
    assert stats["promo_lines_removed_count"] >= 1


def test_preprocess_keeps_dialogue_without_url() -> None:
    raw = "\n".join(
        [
            "— Sign up?",
            "Normal line continues.",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "Sign up?" in cleaned


def test_preprocess_fixes_ocr_spacing() -> None:
    raw = "\n".join(
        [
            "W E RE FINALLY here.",
            "L ET US RETURN to the camp.",
            "F IRST OFF we should go.",
            "S OMETIME it happens.",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "WERE FINALLY here." in cleaned
    assert "LET US RETURN to the camp." in cleaned
    assert "FIRST OFF" in cleaned
    assert "SOMETIME" in cleaned


def test_preprocess_fixes_spaced_caps_and_hyphen_wraps() -> None:
    raw = "\n".join(
        [
            "F IRST OFF— we go.",
            "S OMETIME the rain falls.",
            "M IMORI TOUKA had arrived.",
            "A NOTHER…DIVINE?",
            "W HAT WAS THAT?",
            "W E CONTINUED onward.",
            "I T WAS ON purpose.",
            "th-",
            "think about it.",
            "Pidgey-",
            "chan waved back.",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "FIRST OFF— we go." in cleaned
    assert "SOMETIME the rain falls." in cleaned
    assert "MIMORI TOUKA had arrived." in cleaned
    assert "ANOTHER…DIVINE?" in cleaned
    assert "WHAT WAS THAT?" in cleaned
    assert "WE CONTINUED onward." in cleaned
    assert "IT WAS ON purpose." in cleaned
    assert "th-think about it." in cleaned
    assert "Pidgey-chan waved back." in cleaned


def test_preprocess_reflows_paragraphs_and_preserves_story_start() -> None:
    raw = "\n".join(
        [
            "Table of Contents",
            "Newsletter",
            "Prologue",
            "“F IRST OFF—I’d like to know if we’re going to cast Freeze on",
            "Kirihara.”",
            "Sogou Ayaka’s face was still buried in Takao Hijiri’s chest, but I was speaking mainly to her. Her shoulders twitched in response and I waited for",
            "her reply. When nothing came, Hijiri spoke in her stead.",
            "Chapter 1",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True, skip_front_matter=False)
    assert "OceanofPDF" not in cleaned
    assert "Newsletter" not in cleaned
    assert cleaned.splitlines()[0].startswith("Prologue")
    assert "FIRST OFF" in cleaned
    assert "Chapter 1" in cleaned
    # reflow should have merged mid-sentence breaks
    assert "waited for her reply" in cleaned
    assert stats["reflow_merges"] >= 1
    import re
    assert len(cleaned.splitlines()) < len(raw.splitlines())


def test_preprocess_removes_soft_hyphen_and_spaced_caps() -> None:
    raw = "\n".join(
        [
            "This is an over\u00adwhelming problem.",
            "F IRST and W EIRD samples should merge.",
            "Normal line.",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert "\u00ad" not in cleaned
    assert "overwhelming" in cleaned
    assert "FIRST" in cleaned
    assert "WEIRD" in cleaned
    assert stats["soft_hyphen_removed"] >= 1
    assert stats["spaced_caps_remaining"] == 0


def test_preprocess_removes_watermark_globally_and_merges_dash_continuation() -> None:
    raw = "\n".join(
        [
            "A normal line before.",
            "OceanofPDF.com ruins this line.",
            "He lost consciousness",
            "—falling asleep on the spot.",
            "Zerobooks appears here too.",
            "Final narrative line after spam.",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert "OceanofPDF" not in cleaned
    assert "Zerobooks" not in cleaned
    assert "Join our Discord" not in cleaned
    assert "A normal line before." in cleaned
    assert "Final narrative line after spam." in cleaned
    assert "He lost consciousness —falling asleep on the spot." in cleaned or "He lost consciousness—falling asleep on the spot." in cleaned
    assert stats["watermarks_remaining"] == 0
    assert len(cleaned.splitlines()) < len(raw.splitlines())


def test_preprocess_fixes_under_merge_and_spam_block() -> None:
    raw = "\n".join(
        [
            "Comes up in",
            "science fiction novels frequently.",
            "",
            "Get the latest news and updates.",
            "Or visit us online:",
            "Story continues normally.",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "Comes up in science fiction novels frequently." in cleaned
    assert "Get the latest news" not in cleaned
    assert "visit us online" not in cleaned
    assert "Story continues normally." in cleaned


def test_preprocess_preserves_scene_separators() -> None:
    raw = "\n".join(["***", "***", "Scene continues.", "***"])
    cleaned = preprocess_text(raw)
    assert cleaned.count("***") >= 2


def test_preprocess_preserves_prologue_header() -> None:
    raw = "\n".join(
        [
            "Table of Contents",
            "Prologue",
            "The book begins here.",
            "Chapter 1",
            "A later line.",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "Prologue" in cleaned
    assert "The book begins here." in cleaned


def test_preprocess_preserves_prologue_body_with_toc() -> None:
    raw = "\n".join(
        [
            "Table of Contents",
            "Prologue",
            "Chapter 1",
            "Newsletter",
            "OceanofPDF.com",
            "",
            "Prologue",
            "",
            "“F IRST OFF—I’d like to know if we’re going to cast Freeze on",
            "Kirihara.”",
            "Sogou Ayaka’s face was still buried in Takao Hijiri’s chest, but I was speaking mainly to her. Her shoulders twitched in response and I waited for",
            "her reply. When nothing came, Hijiri spoke in her stead.",
            "“Right… I agree that matter is one that we should discuss in short order.”",
        ]
    )
    cleaned = preprocess_text(raw, skip_front_matter=False)
    assert "FIRST OFF" in cleaned
    assert "Kirihara" in cleaned
    assert "Table of Contents" not in cleaned
    assert "OceanofPDF" not in cleaned
    assert cleaned.splitlines()[0].startswith("Prologue") or "FIRST OFF" in cleaned.splitlines()[0]


class FakeLLMBackend:
    def __init__(self, *args, **kwargs):
        self.backend = "fake"
        self.model = "fake"
        self.temperature = kwargs.get("temperature")
        self.num_predict = kwargs.get("num_predict")
        self.repeat_penalty = kwargs.get("repeat_penalty")

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(text="### TEXTO_TRADUZIDO_INICIO\nTexto limpo.\n### TEXTO_TRADUZIDO_FIM", latency=0.01)


def test_run_translate_preprocess_cleans_noise(monkeypatch, tmp_path: Path) -> None:
    import tradutor.main as main  # noqa: WPS433

    sample_pdf = tmp_path / "sample.pdf"
    sample_pdf.write_text("dummy", encoding="utf-8")

    noisy_text = "\n".join(
        [
            "OceanofPDF.com",
            "Normal story paragraph here.",
            "Table of Contents",
            "Prologue",
            "Chapter 1",
            "Afterword",
            "Another story line.",
            "Sign up for our!",
            "gomanga.com/",
        ]
    )

    monkeypatch.setattr(main, "extract_pdf_text", lambda path, logger: noisy_text)
    monkeypatch.setattr(main, "LLMBackend", FakeLLMBackend)
    monkeypatch.setattr(main, "translate_document", lambda pdf_text, backend, cfg, logger, **kwargs: "texto traduzido")

    cfg = AppConfig(data_dir=tmp_path, output_dir=tmp_path)
    logger = setup_logging()

    args = types.SimpleNamespace(
        command="traduz",
        input=None,
        backend="fake",
        model="fake-model",
        num_predict=32,
        no_refine=True,
        resume=False,
        use_glossary=False,
        manual_glossary=None,
        parallel=1,
        preprocess_advanced=False,
        cleanup_before_refine=None,
        debug_chunks=False,
        debug=True,
        request_timeout=30,
        use_desquebrar=False,
        desquebrar_backend="fake",
        desquebrar_model="fake-desq",
        desquebrar_temperature=0.1,
        desquebrar_chunk_chars=256,
        desquebrar_num_predict=64,
        desquebrar_repeat_penalty=1.0,
        translate_allow_adaptation=False,
        split_by_sections=False,
        fail_on_chunk_error=False,
        pdf_enabled=False,
        skip_front_matter=False,
    )

    main.run_translate(args, cfg, logger)

    run_root = cfg.output_dir / "debug_runs" / "sample"
    runs = sorted(run_root.iterdir())
    assert runs, "debug run should exist"
    run_dir = runs[-1]
    preprocessed_path = run_dir / "10_preprocess" / "sample_preprocessed.md"
    preprocessed = read_text(preprocessed_path)
    assert "OceanofPDF" not in preprocessed
    assert "gomanga.com" not in preprocessed
    assert "Table of Contents" not in preprocessed
    assert "Sign up for" not in preprocessed
    assert "Normal story paragraph" in preprocessed
    assert "Another story line." in preprocessed
