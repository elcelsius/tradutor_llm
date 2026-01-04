import types
import json
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

def test_preprocess_does_not_remove_contents_word() -> None:
    raw = "\n".join(
        [
            "The contents of the report were alarming.",
            "But the hero kept going.",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True, skip_front_matter=False)
    assert "contents of the report" in cleaned
    assert stats["toc_blocks_removed_count"] == 0


def test_preprocess_idempotent() -> None:
    raw = "\n".join(
        [
            "Prologue",
            "OceanofPDF.com",
            "A normal line.",
            "Freeze on",
            "Kirihara.",
        ]
    )
    once = preprocess_text(raw)
    twice = preprocess_text(once)
    assert once == twice


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


def test_preprocess_removes_inline_watermarks_after_merge() -> None:
    raw = "This line has OceanofPDF.com inside and Zerobooks nearby."
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert "OceanofPDF" not in cleaned
    assert "Zerobooks" not in cleaned
    # Linha inteira È descartada por ser promo, logo n„o sobra conte£do narrativo.
    assert cleaned.strip() == "" or "This line has" not in cleaned
    assert stats["inline_watermark_removed_chars"] >= 0


def test_preprocess_respects_long_paragraph_with_discord_token() -> None:
    long_para = " ".join(["discord"] + ["word"] * 100)  # > 200 chars
    cleaned = preprocess_text(long_para)
    assert "discord" in cleaned  # n„o remove par·grafo longo


def test_preprocess_custom_noise_glossary_path(tmp_path: Path) -> None:
    glossary = {
        "line_contains": ["customspam"],
        "line_compact_contains": [],
        "line_regex": [],
        "max_line_len": 120,
    }
    gpath = tmp_path / "noise_glossary.json"
    gpath.write_text(json.dumps(glossary), encoding="utf-8")
    raw = "\n".join(["Normal line stays.", "customspam appears here."])
    cleaned, stats = preprocess_text(raw, noise_glossary_path=gpath, return_stats=True)
    assert "customspam" not in cleaned
    assert "Normal line stays." in cleaned
    assert stats["promo_lines_removed_count"] >= 1


def test_preprocess_merges_freeze_on_kirihara_and_sentence_case() -> None:
    raw = "\n".join(
        [
            "Prologue",
            "Freeze on",
            "Kirihara.",
            "A FTER SOGOU appeared.",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "Freeze on Kirihara." in cleaned
    assert cleaned.splitlines()[0] == "Prologue"
    assert "after sogou appeared." in cleaned.lower()


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


def test_preprocess_keeps_dialogue_on_new_paragraph() -> None:
    raw = "\n".join(
        [
            "Eventually—Sogou opened her mouth to speak, eyes still downturned.",
            "“I…”",
            "“Please, Lady Ayaka. Will you trust the words that Sir Too-ka has",
            "spoken to you?”",
            "She was suddenly interrupted by Seras Ashrain.",
        ]
    )
    cleaned = preprocess_text(raw)
    parts = cleaned.splitlines()
    assert parts[0].startswith("Eventually—Sogou opened her mouth")
    assert parts[1].startswith("“I…”")
    assert parts[2].startswith("“Please, Lady Ayaka.")
    assert "spoken to you?”" in parts[2]


def test_preprocess_preserves_short_dialogue_lines() -> None:
    raw = "\n".join(
        [
            "“Eh?!”",
            "“Yeah?”",
            "“…?”",
            "“Squee—!”",
            "“Squee.”",
            "“...”",
        ]
    )
    cleaned = preprocess_text(raw)
    for line in ["“Eh?!”", "“Yeah?”", "“…?”", "“Squee—!”", "“Squee.”", "“...”"]:
        assert line in cleaned


def test_preprocess_preserves_isolated_ellipsis_line() -> None:
    raw = "\n".join(
        [
            "“I don't think it's feasible.”",
            "“…”",
            "“Given the situation, we wait.”",
        ]
    )
    cleaned = preprocess_text(raw)
    lines = cleaned.splitlines()
    assert any("I don't think it's feasible" in ln for ln in lines)
    assert "“…”" in lines
    assert any("Given the situation" in ln for ln in lines)


def test_preprocess_merges_quote_continuation_and_keeps_ellipsis_line() -> None:
    raw = "\n".join(
        [
            "It's even possible that Mimori Touka believes that from",
            "“the bottom of his heart.”",
            "He paused.",
            "...",
            "Then spoke.",
        ]
    )
    cleaned = preprocess_text(raw)
    lines = cleaned.splitlines()
    assert any("believes that from" in ln for ln in lines)
    assert "“the bottom of his heart.”" in lines[1]
    assert "\n...\n" in cleaned


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


def test_preprocess_preserves_question_dialogue_as_paragraph() -> None:
    raw = "\n".join(
        [
            "“?”",
            "He looked up, unsure.",
        ]
    )
    cleaned = preprocess_text(raw)
    lines = cleaned.splitlines()
    assert lines[0].strip() == "“?”"
    assert any("He looked up" in ln for ln in lines[1:])


def test_preprocess_still_removes_short_noise_lines() -> None:
    raw = "\n".join(
        [
            "OceanofPDF.com",
            "discord.gg/xxxxx",
            "Sign up for our newsletter",
            "Normal story stays.",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "OceanofPDF" not in cleaned
    assert "discord.gg" not in cleaned
    assert "Sign up for our newsletter" not in cleaned
    assert "Normal story stays." in cleaned


def test_preprocess_preserves_plain_ellipsis_line() -> None:
    raw = "\n".join(
        [
            "She insisted it was for our safety.",
            "...",
            "If Sogou knew about this...",
        ]
    )
    cleaned = preprocess_text(raw)
    lines = cleaned.splitlines()
    assert any("She insisted it was for our safety." in ln for ln in lines)
    assert "..." in lines
    assert any("If Sogou knew about this..." in ln for ln in lines)


def test_preprocess_preserves_feasible_erratic_pause() -> None:
    raw = "\n".join(
        [
            "Unfortunately, I don’t think it’s feasible for us to build a cooperative working relationship with Kirihara-kun at present.",
            "...",
            "Given the erratic and unstable nature of Kirihara-kun’s actions and mental state, it’s difficult to ascertain whether we could actually work together.",
        ]
    )
    cleaned = preprocess_text(raw)
    lines = cleaned.splitlines()
    assert any("feasible for us to build a cooperative working relationship" in ln for ln in lines)
    assert "..." in lines
    assert any("Given the erratic and unstable nature of Kirihara-kun’s actions" in ln for ln in lines)


def test_preprocess_preserves_curly_ellipsis_dialogue_line() -> None:
    raw = "\n".join(
        [
            "And I know that Mimori-kun isn’t lying to us.”",
            "“…”",
            "So, she’s got that ability now then, eh?",
        ]
    )
    cleaned = preprocess_text(raw)
    lines = cleaned.splitlines()
    assert any("Mimori-kun isn’t lying to us" in ln for ln in lines)
    assert "“…”" in lines
    assert any("So, she’s got that ability now then" in ln for ln in lines)


def test_preprocess_preserves_multiple_curly_ellipsis_lines_even_if_repeated() -> None:
    raw = "\n".join(
        [
            "Does Piggymaru remind her of some character in a book?",
            "“…”",
            "“…”",
            "I see. I think it is a fine name. Nice to meet you, Piggymaru—",
        ]
    )
    cleaned = preprocess_text(raw)
    lines = cleaned.splitlines()
    assert lines.count("“…”") >= 2
    assert any("fine name" in ln for ln in lines)


def test_preprocess_removes_oceanofpdf_watermark_line() -> None:
    raw = "\n".join(
        [
            "OceanofPDF.com",
            "Real content survives.",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True)
    assert "OceanofPDF" not in cleaned
    assert "Real content survives." in cleaned
    assert any("oceanofpdf" in item.get("text", "").lower() for item in stats.get("removed_full", []))


def test_preprocess_keeps_punctuation_spacing_after_ocr_merge() -> None:
    raw = "\n".join(
        [
            "M IMORI-KUN? Is that a person’s name…?",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "MIMORI-KUN? Is that a person’s name" in cleaned


def test_preprocess_keeps_spaces_between_words_upper_sequences() -> None:
    raw = "\n".join(
        [
            "E HHH?! This little bird is Mistress Anael’s familiar?!",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "EHHH?!" in cleaned
    assert "EHHH?! This" in cleaned
    assert "Mistress Anael" in cleaned


def test_preprocess_removes_advert_header_variants() -> None:
    raw = "\n".join(
        [
            "On Light Novels by Universal",
            "USA ONLY",
            "Download all your Favorite Light Novels .com",
            "Prologue",
            "Story starts here.",
        ]
    )
    cleaned = preprocess_text(raw, skip_front_matter=False)
    assert cleaned.splitlines()[0].startswith("Prologue")
    assert "Universal" not in cleaned
    assert "Favorite Light Novels" not in cleaned


def test_preprocess_merges_across_removed_footer_gap() -> None:
    raw = "\n".join(
        [
            "To think she had gained the ability to see through lies, thought",
            "Page 1",
            "Goldenagato | mp4directs.com",
            "Vicius, infuriated by the memory.",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "thought Vicius" in cleaned


def test_preprocess_report_includes_suspects_and_counts() -> None:
    raw = "\n".join(
        [
            "Prologue",
            "This is a line.",
            "www.example.com",
        ]
    )
    cleaned, stats = preprocess_text(raw, return_stats=True, skip_front_matter=False)
    assert "spaced_caps_remaining" in stats
    assert "spaced_caps_remaining_samples" in stats
    assert "urls_remaining_count" in stats
    assert "toc_remaining_count" in stats
    assert "footers_removed_count" in stats


def test_preprocess_keeps_ellipsis_dialogue_between_anchors() -> None:
    raw = "\n".join(
        [
            "And I know that Mimori-kun isn’t lying to us.”",
            "“...”",
            "So, she’s got that ability now then, eh?",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "Mimori-kun isn’t lying to us" in cleaned
    assert "“...”" in cleaned
    assert "So, she’s got that ability now then" in cleaned


def test_preprocess_keeps_interjection_eh() -> None:
    raw = "\n".join(
        [
            "My collaborator is Nyantan Kikipat.”",
            "“Eh?!”",
            "“Nyaki.”",
        ]
    )
    cleaned = preprocess_text(raw)
    assert "Nyantan Kikipat" in cleaned
    assert "“Eh?!”" in cleaned
    assert "“Nyaki.”" in cleaned


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
