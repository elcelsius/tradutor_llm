import json
import logging
from pathlib import Path

from tradutor.bilingual_review import (
    detect_bilingual_review_focuses,
    review_translation_chunk,
)
from tradutor.cache_utils import set_cache_base_dir
from tradutor.config import AppConfig
from tradutor.translate import translate_document


class _SuggestionBackend:
    backend = "stub"
    model = "review-stub"
    num_predict = 100
    temperature = 0.1
    repeat_penalty = 1.0

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str):
        self.calls += 1
        assert "ORIGINAL EM INGLES" in prompt
        text = (
            "```json\n"
            '{"correcoes":[{"original":"Eu não pensei que alguém me notava",'
            '"substituicao":"Eu não achei que alguém reparasse em mim",'
            '"motivo":"gramatica e sentido"}]}'
            "\n```"
        )
        return type("Resp", (), {"text": text})


class _QuoteChangingSuggestionBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"“Olá”",'
            '"substituicao":"Olá",'
            '"motivo":"nao deve alterar marcacao"}]}'
        )
        return type("Resp", (), {"text": text})


class _OptionalProperNameArticleBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"as palavras da Hijiri",'
            '"substituicao":"as palavras de Hijiri",'
            '"motivo":"regencia"}]}'
        )
        return type("Resp", (), {"text": text})


class _GlossaryNameChangingBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"observou Hijiri Takao.",'
            '"substituicao":"observou Hijiri.",'
            '"motivo":"consistencia"}]}'
        )
        return type("Resp", (), {"text": text})


class _SingleTokenSynonymBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"não têm Aprimoramento Anti-Divino",'
            '"substituicao":"não possuem Aprimoramento Anti-Divino",'
            '"motivo":"fluidez"}]}'
        )
        return type("Resp", (), {"text": text})


class _ArticleBeforeNameBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"mais forte que a Vicius",'
            '"substituicao":"mais forte que Vicius",'
            '"motivo":"estilo"}]}'
        )
        return type("Resp", (), {"text": text})


class _HonorificArticleBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"falou com a Sogou-san",'
            '"substituicao":"falou com Sogou-san",'
            '"motivo":"estilo"}]}'
        )
        return type("Resp", (), {"text": text})


class _UnknownNameBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"Sogou e Kirihara",'
            '"substituicao":"Sogou e Kirarina",'
            '"motivo":"consistencia"}]}'
        )
        return type("Resp", (), {"text": text})


class _KnownGenderChangingBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"Sogou claramente foi afetada",'
            '"substituicao":"Sogou claramente foi afetado",'
            '"motivo":"concordancia"}]}'
        )
        return type("Resp", (), {"text": text})


class _MissingReasonBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"ela foi embora",'
            '"substituicao":"ela partiu","motivo":""}]}'
        )
        return type("Resp", (), {"text": text})


class _BrainwashCalqueBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"foi lavada cerebral",'
            '"substituicao":"foi lavada o cérebro",'
            '"motivo":"expressao"}]}'
        )
        return type("Resp", (), {"text": text})


class _LiteralMoveBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"as forças se moveram",'
            '"substituicao":"as forças fizeram seu movimento",'
            '"motivo":"sentido"}]}'
        )
        return type("Resp", (), {"text": text})


class _UngrammaticalSubjunctiveBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"facilita para os outros tingi-la",'
            '"substituicao":"facilita que os outros a tingem",'
            '"motivo":"gramatica"}]}'
        )
        return type("Resp", (), {"text": text})


class _FixedSuggestionBackend(_SuggestionBackend):
    def __init__(self, original: str, replacement: str, reason: str = "gramatica") -> None:
        super().__init__()
        self.original = original
        self.replacement = replacement
        self.reason = reason

    def generate(self, prompt: str):
        self.calls += 1
        text = json.dumps(
            {
                "correcoes": [
                    {
                        "original": self.original,
                        "substituicao": self.replacement,
                        "motivo": self.reason,
                    }
                ]
            },
            ensure_ascii=False,
        )
        return type("Resp", (), {"text": text})


class _PeriphrasticPresentPerfectBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"Vicius se moveu!",'
            '"substituicao":"Vicius tem se movido!",'
            '"motivo":"tempo verbal"}]}'
        )
        return type("Resp", (), {"text": text})


class _PeriphrasticPastPerfectBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"Vicius se moveu!",'
            '"substituicao":"Vicius tinha se movido!",'
            '"motivo":"tempo verbal"}]}'
        )
        return type("Resp", (), {"text": text})


class _SubordinateClauseBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"Pode ser uma bênção alguém do seu '
            'nível ter sido mandado.",'
            '"substituicao":"Pode ser uma bênção que alguém do seu nível '
            'tenha sido mandado.",'
            '"motivo":"oracao subordinada sem conectivo e modo verbal"}]}'
        )
        return type("Resp", (), {"text": text})


class _DialogueTreatmentBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"Apressa-te e ajoelha-te.",'
            '"substituicao":"Apresse-se e ajoelhe-se.",'
            '"motivo":"uniformizar tratamento"}]}'
        )
        return type("Resp", (), {"text": text})


class _ShortPhraseReorderBackend(_SuggestionBackend):
    def generate(self, prompt: str):
        self.calls += 1
        text = (
            '{"correcoes":[{"original":"vento caverna",'
            '"substituicao":"caverna de vento",'
            '"motivo":"ordem das palavras"}]}'
        )
        return type("Resp", (), {"text": text})


class _TranslationBackend:
    backend = "stub"
    model = "translate-stub"
    num_predict = 100
    temperature = 0.1
    repeat_penalty = 1.0

    def generate(self, prompt: str):
        text = (
            "### TEXTO_TRADUZIDO_INICIO\n"
            "“Eu não pensei que alguém me notava.”\n"
            "### TEXTO_TRADUZIDO_FIM"
        )
        return type("Resp", (), {"text": text})


class _RetryTranslationBackend(_TranslationBackend):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str):
        self.calls += 1
        if self.calls == 1:
            text = (
                "### TEXTO_TRADUZIDO_INICIO\n"
                "“I have no desire to die.”\n"
                "### TEXTO_TRADUZIDO_FIM"
            )
        else:
            text = (
                "### TEXTO_TRADUZIDO_INICIO\n"
                "“Não tenho vontade de morrer.”\n"
                "### TEXTO_TRADUZIDO_FIM"
            )
        return type("Resp", (), {"text": text})


def test_bilingual_review_applies_exact_minimal_change_and_uses_cache(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    backend = _SuggestionBackend()
    kwargs = {
        "source_text": '“I did not think anyone paid attention to me.”',
        "translated_text": "“Eu não pensei que alguém me notava.”",
        "backend": backend,
        "logger": logging.getLogger("bilingual-review-test"),
    }

    first = review_translation_chunk(**kwargs)
    second = review_translation_chunk(**kwargs)

    assert first.attempted and first.changed
    assert "Eu não achei que alguém reparasse em mim" in first.text
    assert first.applied_changes
    assert second.used_cache
    assert backend.calls == 1


def test_bilingual_review_rejects_change_that_alters_dialogue_markers(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "“Olá”"
    result = review_translation_chunk(
        source_text='“Hello”',
        translated_text=current,
        backend=_QuoteChangingSuggestionBackend(),
        logger=logging.getLogger("bilingual-review-quote-test"),
    )

    assert result.attempted
    assert not result.changed
    assert result.text == current
    assert result.rejected_changes
    assert result.rejected_changes[0]["rejeitada_por"] == "review_changed_quote_count"


def test_bilingual_review_rejects_optional_article_before_proper_name(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Kashima se agitou, como se as palavras da Hijiri ganhassem vida."
    result = review_translation_chunk(
        source_text="Kashima flailed as if Hijiri's words had come to life.",
        translated_text=current,
        backend=_OptionalProperNameArticleBackend(),
        logger=logging.getLogger("bilingual-review-article-test"),
    )

    assert result.attempted
    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_optional_proper_name_article"


def test_bilingual_review_rejects_changes_to_protected_glossary_name(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Ela concordou, observou Hijiri Takao."
    result = review_translation_chunk(
        source_text="She agreed, noted Hijiri.",
        translated_text=current,
        backend=_GlossaryNameChangingBackend(),
        logger=logging.getLogger("bilingual-review-glossary-name-test"),
        glossary_terms=[
            {
                "key": "Takao Hijiri",
                "pt": "Takao Hijiri",
                "source_aliases": ["Hijiri Takao", "Hijiri"],
                "locked": True,
                "type": "personagem",
            }
        ],
    )

    assert result.attempted
    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_changed_protected_glossary_form"


def test_bilingual_review_rejects_single_token_synonym(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "Elas não têm Aprimoramento Anti-Divino."
    result = review_translation_chunk(
        source_text="Those without anti-divine enhancement.",
        translated_text=current,
        backend=_SingleTokenSynonymBackend(),
        logger=logging.getLogger("bilingual-review-synonym-test"),
    )

    assert result.attempted
    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_single_token_synonym"


def test_bilingual_review_rejects_article_style_change_before_proper_name(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Ela pode ser mais forte que a Vicius."
    result = review_translation_chunk(
        source_text="She might be stronger than Vicius.",
        translated_text=current,
        backend=_ArticleBeforeNameBackend(),
        logger=logging.getLogger("bilingual-review-article-name-test"),
    )

    assert result.attempted
    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_optional_proper_name_article"


def test_bilingual_review_rejects_article_style_change_before_honorific(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Itsuki falou com a Sogou-san."
    result = review_translation_chunk(
        source_text="Itsuki spoke to Sogou-san.",
        translated_text=current,
        backend=_HonorificArticleBackend(),
        logger=logging.getLogger("bilingual-review-honorific-article-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_optional_proper_name_article"
    )


def test_bilingual_review_rejects_unknown_proper_name(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "Sogou e Kirihara permaneceram escondidos."
    result = review_translation_chunk(
        source_text="Sogou and Kirihara stayed hidden.",
        translated_text=current,
        backend=_UnknownNameBackend(),
        logger=logging.getLogger("bilingual-review-unknown-name-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_introduced_unknown_proper_name"
    )


def test_bilingual_review_rejects_known_glossary_gender_regression(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Sogou claramente foi afetada pelas palavras."
    result = review_translation_chunk(
        source_text="Sogou was clearly affected by the words.",
        translated_text=current,
        backend=_KnownGenderChangingBackend(),
        logger=logging.getLogger("bilingual-review-gender-test"),
        glossary_terms=[
            {
                "key": "Sogou Ayaka",
                "pt": "Sogou Ayaka",
                "aliases": ["Sogou"],
                "category": "personagem",
                "gender": "feminino",
            }
        ],
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_changed_known_gender"


def test_bilingual_review_rejects_suggestion_without_reason(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "Ela foi embora."
    result = review_translation_chunk(
        source_text="She left.",
        translated_text=current,
        backend=_MissingReasonBackend(),
        logger=logging.getLogger("bilingual-review-missing-reason-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_missing_reason"


def test_bilingual_review_rejects_brainwash_calque(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "Sogou foi lavada cerebral pela Deusa."
    result = review_translation_chunk(
        source_text="Sogou was brainwashed by the Goddess.",
        translated_text=current,
        backend=_BrainwashCalqueBackend(),
        logger=logging.getLogger("bilingual-review-brainwash-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_brainwash_calque"


def test_bilingual_review_rejects_literal_made_move_calque(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "As forças se moveram."
    result = review_translation_chunk(
        source_text="The forces have made their move.",
        translated_text=current,
        backend=_LiteralMoveBackend(),
        logger=logging.getLogger("bilingual-review-move-calque-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_literal_move_calque"


def test_bilingual_review_rejects_ungrammatical_tingir_subjunctive(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Sua pureza facilita para os outros tingi-la."
    result = review_translation_chunk(
        source_text="Her purity makes it easy for others to dye her.",
        translated_text=current,
        backend=_UngrammaticalSubjunctiveBackend(),
        logger=logging.getLogger("bilingual-review-subjunctive-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_ungrammatical_subjunctive"
    )


def test_bilingual_review_rejects_inline_optional_article_change(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "A Sogou começou a chorar novamente."
    result = review_translation_chunk(
        source_text="Sogou started crying again.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "A Sogou começou a chorar novamente.",
            "Sogou começou a chorar novamente.",
            "estilo",
        ),
        logger=logging.getLogger("bilingual-review-inline-article-test"),
    )

    assert not result.changed
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_optional_proper_name_article"
    )


def test_bilingual_review_allows_simple_article_agreement_fix(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "O criatura avançou."
    result = review_translation_chunk(
        source_text="The creature advanced.",
        translated_text=current,
        backend=_FixedSuggestionBackend("O criatura", "A criatura"),
        logger=logging.getLogger("bilingual-review-article-agreement-test"),
    )

    assert result.changed
    assert result.text == "A criatura avançou."


def test_bilingual_review_rejects_collective_singular_style_change(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "A maioria das máscaras que usamos é adotada por escolha própria."
    result = review_translation_chunk(
        source_text="Most of the masks we use are chosen.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "A maioria das máscaras que usamos é adotada",
            "A maioria das máscaras que usamos são adotadas",
        ),
        logger=logging.getLogger("bilingual-review-collective-test"),
    )

    assert not result.changed
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_collective_singular_agreement"
    )


def test_bilingual_review_rejects_brainwash_meaning_loss(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "Sogou foi lavada cerebral pela Deusa."
    result = review_translation_chunk(
        source_text="Sogou was brainwashed by the Goddess.",
        translated_text=current,
        backend=_FixedSuggestionBackend("foi lavada cerebral", "foi manipulada"),
        logger=logging.getLogger("bilingual-review-brainwash-meaning-test"),
    )

    assert not result.changed
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_brainwash_semantic_loss"
    )


def test_bilingual_review_allows_leading_capitalization_typo_fix(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "Ela sorriu. Ahora então, ajoelhe-se."
    result = review_translation_chunk(
        source_text="She smiled. Now then, kneel.",
        translated_text=current,
        backend=_FixedSuggestionBackend("Ahora então", "Agora então", "ortografia"),
        logger=logging.getLogger("bilingual-review-leading-capital-test"),
    )

    assert result.changed
    assert result.text == "Ela sorriu. Agora então, ajoelhe-se."


def test_bilingual_review_rejects_feminine_member_to_masculine_change(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Você também é uma membro importante."
    result = review_translation_chunk(
        source_text="You are an important member too.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "Você também é uma membro importante",
            "Você também é um membro importante",
        ),
        logger=logging.getLogger("bilingual-review-member-gender-test"),
    )

    assert not result.changed
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_changed_feminine_member"
    )


def test_bilingual_review_allows_mixed_group_plural_agreement(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Seras, Munin e Itsuki estavam afastadas."
    result = review_translation_chunk(
        source_text="Seras, Munin, and Itsuki were standing apart.",
        translated_text=current,
        backend=_FixedSuggestionBackend("estavam afastadas", "estavam afastados"),
        logger=logging.getLogger("bilingual-review-mixed-group-test"),
        glossary_terms=[
            {
                "key": "Munin",
                "pt": "Munin",
                "category": "personagem",
                "gender": "feminino",
            }
        ],
    )

    assert result.changed
    assert result.text == "Seras, Munin e Itsuki estavam afastados."


def test_bilingual_review_rejects_periphrastic_present_perfect_calque(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Vicius se moveu!"
    result = review_translation_chunk(
        source_text="Vicius has moved!",
        translated_text=current,
        backend=_PeriphrasticPresentPerfectBackend(),
        logger=logging.getLogger("bilingual-review-aspect-test"),
    )

    assert result.attempted
    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_periphrastic_perfect_calque"


def test_bilingual_review_rejects_past_perfect_calque(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "Vicius se moveu!"
    result = review_translation_chunk(
        source_text="Vicius has moved!",
        translated_text=current,
        backend=_PeriphrasticPastPerfectBackend(),
        logger=logging.getLogger("bilingual-review-past-aspect-test"),
    )

    assert result.rejected_changes[0]["rejeitada_por"] == "review_periphrastic_perfect_calque"


def test_bilingual_review_rejects_unpronominal_recover_from_calque(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Esse é o tipo de falha que podemos recuperar."
    result = review_translation_chunk(
        source_text="This is the kind of failure we can recover from.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "falha que podemos recuperar",
            "falha de que podemos recuperar",
            "regência",
        ),
        logger=logging.getLogger("bilingual-review-recover-from-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_recover_from_calque"


def test_bilingual_review_rejects_lembrei_register_formalization(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Lembrei das palavras da Seras."
    result = review_translation_chunk(
        source_text="I remembered Seras's words.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "Lembrei das palavras da Seras",
            "Lembrei-me das palavras da Seras",
            "regência",
        ),
        logger=logging.getLogger("bilingual-review-lembrei-style-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_lembrei_registro"


def test_bilingual_review_rejects_lembrei_direct_object_formalization(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Lembrei das palavras da Seras."
    result = review_translation_chunk(
        source_text="I remembered Seras's words.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "Lembrei das palavras da Seras",
            "Lembrei as palavras da Seras",
            "regência",
        ),
        logger=logging.getLogger("bilingual-review-lembrei-direct-object-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_lembrei_registro"


def test_bilingual_review_rejects_colloquial_direct_object_formalization(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Nós estudamos ele com cuidado."
    result = review_translation_chunk(
        source_text="We studied him carefully.",
        translated_text=current,
        backend=_FixedSuggestionBackend("estudamos ele", "o estudamos"),
        logger=logging.getLogger("bilingual-review-direct-object-style-test"),
    )

    assert not result.changed
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_colloquial_register"
    )


def test_bilingual_review_rejects_colloquial_thanks_formalization(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Quero agradecer você por isso."
    result = review_translation_chunk(
        source_text="I want to thank you for that.",
        translated_text=current,
        backend=_FixedSuggestionBackend("agradecer você", "agradecer-lhe"),
        logger=logging.getLogger("bilingual-review-thanks-style-test"),
    )

    assert not result.changed
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_colloquial_register"
    )


def test_bilingual_review_rejects_colloquial_deixar_eu_formalization(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Deixar eu me vingar é tudo o que peço."
    result = review_translation_chunk(
        source_text="Let me get my revenge. That is all I ask.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "Deixar eu me vingar", "Deixar que eu me vingue"
        ),
        logger=logging.getLogger("bilingual-review-deixar-style-test"),
    )

    assert not result.changed
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_colloquial_register"
    )


def test_bilingual_review_rejects_bad_recovery_regency_and_subjunctive(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Assim que ela recuperar a consciência, vamos rezar para que ela se recupere."
    result = review_translation_chunk(
        source_text="Once she regains consciousness, let us pray that she recovers.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "recuperar a consciência, vamos rezar para que ela se recupere",
            "recuperar-se da consciência, vamos rezar para que ela se recuperar",
        ),
        logger=logging.getLogger("bilingual-review-recovery-regency-test"),
    )

    assert not result.changed
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_recover_consciousness"
    )


def test_bilingual_review_rejects_duplicate_recovery_reflexive(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "Ele parece estar se recuperando mais rápido."
    result = review_translation_chunk(
        source_text="He appears to be recovering faster.",
        translated_text=current,
        backend=_FixedSuggestionBackend("se recuperando", "se recuperando-se"),
        logger=logging.getLogger("bilingual-review-duplicate-recovery-test"),
    )

    assert not result.changed
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_duplicate_recover_reflexive"
    )


def test_bilingual_review_allows_generic_location_title_word_to_change(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Ela tem sido longe de inativa."
    result = review_translation_chunk(
        source_text="She has been far from inactive.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "Ela tem sido longe de inativa",
            "Ela está longe de ser inativa",
            "regência",
        ),
        logger=logging.getLogger("bilingual-review-generic-location-title-test"),
        glossary_terms=[
            {
                "key": "State of Yonato",
                "pt": "Estado de Yonato",
                "category": "local",
                "locked": True,
                "aliases": ["Yonato"],
            }
        ],
    )

    assert result.changed
    assert result.text == "Ela está longe de ser inativa."


def test_bilingual_review_rejects_unanchored_rewrite(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "As pessoas são congeladas quando um for desenvolvido."
    result = review_translation_chunk(
        source_text="People are frozen until a treatment is developed.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "quando um for desenvolvido",
            "ver através do engano",
            "sentido",
        ),
        logger=logging.getLogger("bilingual-review-unanchored-rewrite-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_unanchored_rewrite"


def test_bilingual_review_rejects_dangling_preposition(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "O choque mental pode levar um tempo para superar."
    result = review_translation_chunk(
        source_text="The mental shock might take her a while to get over.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "para superar.",
            "para ela se recuperar de.",
            "regência",
        ),
        logger=logging.getLogger("bilingual-review-dangling-preposition-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_dangling_preposition"


def test_bilingual_review_rejects_invalid_brainwash_auxiliary(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "Só se alguém fosse totalmente lavado pelo cérebro conseguiria confiar nela."
    result = review_translation_chunk(
        source_text="Only someone totally brainwashed could trust her.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "fosse totalmente lavado pelo cérebro",
            "fosse sofreu lavagem cerebral",
            "gramática",
        ),
        logger=logging.getLogger("bilingual-review-brainwash-auxiliary-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_brainwash_auxiliary"


def test_bilingual_review_rejects_duplicate_joined_together(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    current = "Sogou e Seras deveriam ter se juntado desde o começo."
    result = review_translation_chunk(
        source_text="Sogou and Seras should have teamed up together from the start.",
        translated_text=current,
        backend=_FixedSuggestionBackend(
            "ter se juntado desde o começo",
            "ter se juntado juntos",
            "fluidez",
        ),
        logger=logging.getLogger("bilingual-review-duplicate-joined-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_duplicate_joined_together"
    )


def test_bilingual_review_accepts_source_aware_subordinate_clause_fix(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    result = review_translation_chunk(
        source_text=(
            "It might be a blessing that someone on your level was sent."
        ),
        translated_text="Pode ser uma bênção alguém do seu nível ter sido mandado.",
        backend=_SubordinateClauseBackend(),
        logger=logging.getLogger("bilingual-review-subordinate-clause-test"),
    )

    assert result.changed
    assert result.text == "Pode ser uma bênção que alguém do seu nível tenha sido mandado."


def test_bilingual_review_rejects_dialogue_treatment_normalization(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Apressa-te e ajoelha-te."
    result = review_translation_chunk(
        source_text="Kneel quickly.",
        translated_text=current,
        backend=_DialogueTreatmentBackend(),
        logger=logging.getLogger("bilingual-review-treatment-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == (
        "review_changed_dialogue_treatment"
    )


def test_bilingual_review_rejects_partial_noun_phrase_reorder(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    current = "Só ouço o uivo de um patético vento caverna!"
    result = review_translation_chunk(
        source_text="All I hear is the howling of a pathetic little wind cave!",
        translated_text=current,
        backend=_ShortPhraseReorderBackend(),
        logger=logging.getLogger("bilingual-review-noun-phrase-test"),
    )

    assert not result.changed
    assert result.text == current
    assert result.rejected_changes[0]["rejeitada_por"] == "review_short_phrase_reorder"


def test_bilingual_review_detects_source_aware_focus_patterns() -> None:
    text = (
        "Na mão direita de Vicius, ela segurava a cabeça. "
        "Pode ser uma bênção alguém do seu nível ter sido mandado."
    )

    focuses = detect_bilingual_review_focuses(text)

    assert len(focuses) == 2
    assert focuses[0].startswith("Na mão direita de Vicius, ela segurava")
    assert focuses[1].startswith("Pode ser uma bênção alguém")


def test_translate_document_runs_bilingual_review_and_writes_metrics(
    tmp_path: Path,
) -> None:
    set_cache_base_dir(tmp_path)
    cfg = AppConfig(
        output_dir=tmp_path,
        max_retries=1,
        split_by_sections=False,
        use_translation_repair=False,
        bilingual_review_after_translate=True,
    )

    result = translate_document(
        pdf_text='“I did not think anyone paid attention to me.”',
        backend=_TranslationBackend(),
        cfg=cfg,
        logger=logging.getLogger("bilingual-review-integration-test"),
        source_slug="sample",
        already_preprocessed=True,
        bilingual_review=True,
        bilingual_review_backend=_SuggestionBackend(),
    )

    assert "Eu não achei que alguém reparasse em mim" in result
    metrics = json.loads(
        (tmp_path / "sample_bilingual_review_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["enabled"]
    assert metrics["changed_chunks"] == 1
    assert metrics["chunks"][0]["applied_changes"]


def test_translation_metrics_accumulate_outer_qa_retries(tmp_path: Path) -> None:
    set_cache_base_dir(tmp_path)
    cfg = AppConfig(
        output_dir=tmp_path,
        max_retries=2,
        split_by_sections=False,
        use_translation_repair=False,
    )

    result = translate_document(
        pdf_text='“I have no desire to die.”',
        backend=_RetryTranslationBackend(),
        cfg=cfg,
        logger=logging.getLogger("translation-attempt-metrics-test"),
        source_slug="attempts",
        already_preprocessed=True,
    )

    assert "Não tenho vontade de morrer" in result
    metrics = json.loads(
        (tmp_path / "attempts_translate_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["chunks"][0]["llm_attempts"] == 2
