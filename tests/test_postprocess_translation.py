import logging
from pathlib import Path

from tradutor.config import AppConfig
from tradutor.postprocess_translation import postprocess_translation
from tradutor.translate import translate_document


class _ParryBackend:
    """Processamento interno auxiliar."""

    def __init__(self) -> None:
        """Processamento interno auxiliar."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 128
        self.temperature = 0.1
        self.repeat_penalty = 1.0

    def generate(self, prompt: str):
        """Processamento interno auxiliar."""
        return type(
            "Resp",
            (),
            {
                "text": "### TEXTO_TRADUZIDO_INICIO\nEle parriu o golpe rapidamente.\n### TEXTO_TRADUZIDO_FIM"
            },
        )


def test_postprocess_fixes_parry_false_cognate(tmp_path: Path) -> None:
    """Processamento interno auxiliar."""
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


class _MixedEnglishArtifactBackend:
    """Processamento interno auxiliar."""

    def __init__(self) -> None:
        """Processamento interno auxiliar."""
        self.backend = "stub"
        self.model = "stub"
        self.num_predict = 128
        self.temperature = 0.1
        self.repeat_penalty = 1.0
        self.calls = 0

    def generate(self, prompt: str):
        """Processamento interno auxiliar."""
        self.calls += 1
        return type(
            "Resp",
            (),
            {
                "text": "### TEXTO_TRADUZIDO_INICIO\n… I-isso? I não me importo.\n### TEXTO_TRADUZIDO_FIM"
            },
        )


def test_pre_qa_postprocess_fixes_mixed_english_artifacts(tmp_path: Path) -> None:
    """Processamento interno auxiliar."""
    cfg = AppConfig(output_dir=tmp_path, split_by_sections=False)
    backend = _MixedEnglishArtifactBackend()
    logger = logging.getLogger("mixed-english-pre-qa")

    result = translate_document(
        pdf_text="… Y-yes? I do not care.",
        backend=backend,
        cfg=cfg,
        logger=logger,
        source_slug="mixed-artifact",
    )

    assert "S-sim? Eu não me importo." in result
    assert backend.calls == 1


def test_pre_qa_postprocess_does_not_turn_full_english_into_hybrid() -> None:
    result = postprocess_translation("I have no desire to die.")

    assert result == "I have no desire to die."


def test_pre_qa_postprocess_adapts_af_slang() -> None:
    """Processamento interno auxiliar."""
    result = postprocess_translation("Ela é super desconfiada AF.")

    assert result == "Ela é super desconfiada pra caramba."


def test_pre_qa_postprocess_translates_single_word_connector() -> None:
    """Processamento interno auxiliar."""
    result = postprocess_translation("Ela é arrogante, though.")

    assert result == "Ela é arrogante, porém."


def test_pre_qa_postprocess_translates_arright_interjection() -> None:
    """Processamento interno auxiliar."""
    result = postprocess_translation("“Arright!” Itsuki sorriu.")

    assert result == "“Beleza!” Itsuki sorriu."


def test_pre_qa_postprocess_removes_hybrid_english_pronoun() -> None:
    """Processamento interno auxiliar."""
    result = postprocess_translation(
        "Ninguém o encarava com raiva — they todos pareciam felizes."
    )

    assert result == "Ninguém o encarava com raiva — todos pareciam felizes."


def test_pre_qa_postprocess_translates_boost() -> None:
    """Processamento interno auxiliar."""
    result = postprocess_translation("Eles precisavam de um boost extra.")

    assert result == "Eles precisavam de um impulso extra."


def test_pre_qa_postprocess_fixes_mixed_pronoun_and_interjection() -> None:
    """Processamento interno auxiliar."""
    result = postprocess_translation(
        "Uau—I aposto que ela consegue. Uhh… sei disso. Uh… também."
    )

    assert result == "Uau—Eu aposto que ela consegue. Ah… sei disso. Ah… também."


def test_pre_qa_postprocess_fixes_english_stutter_before_qa() -> None:
    """Processamento interno auxiliar."""
    result = postprocess_translation("Y-you divindades podem ser interessantes.")

    assert result == "V-vocês, divindades, podem ser interessantes."


def test_pre_qa_postprocess_fixes_malformed_portuguese_divinity_stutter() -> None:
    result = postprocess_translation("V-você divinos podem ser interessantes.")

    assert result == "V-vocês, divindades, podem ser interessantes."


def test_pre_qa_postprocess_fixes_embedded_connectors_without_blind_gender_swap() -> None:
    result = postprocess_translation(
        "And assim seguimos, or pelo menos tentamos, but esses eucaristos vencerão."
    )

    assert result == "E assim seguimos, ou pelo menos tentamos, mas esses eucaristos vencerão."


def test_pre_qa_postprocess_corrects_carriage_artifacts_only_for_carriage_source() -> None:
    source = "They returned to the carriage and waited inside that carriage."
    result = postprocess_translation(
        "Voltaram para o carroça e esperaram dentro daquele carro.", en_text=source
    )
    untouched = postprocess_translation("O carro estava vazio.", en_text="The car was empty.")

    assert result == "Voltaram para a carruagem e esperaram dentro daquela carruagem."
    assert untouched == "O carro estava vazio."


def test_pre_qa_postprocess_fixes_recurrent_literal_artifacts() -> None:
    result = postprocess_translation(
        "A algum momento depois de chegar, Convincentemente Nyantan a confiar nela "
        "foi impressionante. Yasu fora um dos vários que pegaram no caminho. "
        "Ela estava tã bonita. Terei a maior dor sobre todas as dimensões e todos os mundos. "
        "Tragarei o maior sofrimento sobre todas as dimensões e todos os mundos. "
        "Forra uma criatura fraca. Os trajes de espadachins voadoras chegaram. "
        "Ah, Pode calar a boca. As divindades mais significativas fossem enviadas. "
        "Virando seu ódio uns contra os outros, ela chutou-a levemente quando ela caiu. "
        "Ela chutou levemente de volta quando ela caiu."
    )

    assert "Em algum momento depois de chegar" in result
    assert "Convencer Nyantan a confiar nela foi impressionante" in result
    assert "Yasu fora um dos vários que eles haviam levado consigo pelo caminho" in result
    assert "tão bonita" in result
    assert "Trarei o maior sofrimento sobre todas as dimensões e todos os mundos" in result
    assert "Tragarei o maior sofrimento" not in result
    assert "Era uma criatura fraca" in result
    assert "trajes de espadachins voadores" in result
    assert "Ah, pode calar a boca" in result
    assert "divindades mais poderosas fossem enviadas" in result
    assert "voltando seu ódio uns contra os outros" in result
    assert "chutou-a de leve de volta para o alto quando ela caiu" in result


def test_pre_qa_postprocess_fixes_literal_matter_only_with_matching_source() -> None:
    result = postprocess_translation(
        "Há uma coisa em mente.",
        en_text="There is one matter on my mind.",
    )
    untouched = postprocess_translation("Há uma coisa em mente.")

    assert result == "Há uma coisa que está me preocupando."
    assert untouched == "Há uma coisa em mente."
