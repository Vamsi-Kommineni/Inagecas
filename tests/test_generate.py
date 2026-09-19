from __future__ import annotations

from answer_service.generate import generate
from factories import FakeGateway, make_evidence
from inagecas_shared.config import get_settings
from inagecas_shared.schemas import AnswerStatus, GenerateRequest, RefusalReason

settings = get_settings()


async def test_refuses_without_evidence():
    result = await generate(FakeGateway(), settings, GenerateRequest(question="q", evidence=[]))
    assert result.status == AnswerStatus.refused
    assert result.refusal_reason == RefusalReason.no_evidence
    assert result.confidence == 0.0


async def test_refuses_below_confidence_threshold():
    request = GenerateRequest(question="q", evidence=[make_evidence(score=0.2)], top_score=0.2)
    result = await generate(FakeGateway("anything"), settings, request)
    assert result.status == AnswerStatus.refused
    assert result.refusal_reason == RefusalReason.low_confidence


async def test_refuses_when_model_signals_insufficient_context():
    request = GenerateRequest(question="q", evidence=[make_evidence(score=0.9)], top_score=0.9)
    result = await generate(FakeGateway("INSUFFICIENT_CONTEXT"), settings, request)
    assert result.status == AnswerStatus.refused
    assert result.refusal_reason == RefusalReason.unsupported


async def test_refuses_when_the_sentinel_is_written_with_a_space():
    """llama3.2:3b writes "INSUFFICIENT CONTEXT"; that is the same refusal."""
    request = GenerateRequest(question="q", evidence=[make_evidence(score=0.9)], top_score=0.9)
    result = await generate(FakeGateway("INSUFFICIENT CONTEXT"), settings, request)
    assert result.status == AnswerStatus.refused
    assert result.refusal_reason == RefusalReason.unsupported
    # No judge call was needed: the model itself declined.
    assert result.support is None


async def test_answers_when_validation_supports_it():
    evidence = [make_evidence(score=0.9), make_evidence(score=0.85, ordinal=1)]
    gateway = FakeGateway(replies=["Paris is the capital [1].", "SUPPORTED"])
    request = GenerateRequest(question="capital?", evidence=evidence, top_score=0.9)
    result = await generate(gateway, settings, request)
    assert result.status == AnswerStatus.answered
    assert result.answer == "Paris is the capital [1]."
    assert result.support == "supported"
    assert result.coverage == 1.0
    assert [c.chunk_id for c in result.citations] == [evidence[0].chunk_id]


async def test_contradicted_answer_is_refused():
    gateway = FakeGateway(replies=["Some claim [1].", "CONTRADICTED"])
    request = GenerateRequest(question="q", evidence=[make_evidence(score=0.9)], top_score=0.9)
    result = await generate(gateway, settings, request)
    assert result.status == AnswerStatus.refused
    assert result.refusal_reason == RefusalReason.contradicted
    assert result.support == "contradicted"


async def test_unsupported_answer_is_refused_despite_high_retrieval():
    gateway = FakeGateway(replies=["Some claim [1].", "UNSUPPORTED"])
    request = GenerateRequest(question="q", evidence=[make_evidence(score=0.9)], top_score=0.9)
    result = await generate(gateway, settings, request)
    assert result.status == AnswerStatus.refused
    assert result.refusal_reason == RefusalReason.unsupported


async def test_a_refused_draft_is_kept_for_the_human_but_not_the_customer():
    gateway = FakeGateway(replies=["Some claim [1].", "UNSUPPORTED"])
    request = GenerateRequest(question="q", evidence=[make_evidence(score=0.9)], top_score=0.9)
    result = await generate(gateway, settings, request)
    assert result.answer is None
    assert result.draft == "Some claim [1]."


async def test_borderline_partial_asks_for_clarification():
    gateway = FakeGateway(
        replies=["Try resetting it [1].\n\nOr reinstall.", "PARTIAL", "Which device are you using?"]
    )
    request = GenerateRequest(
        question="it doesn't work", evidence=[make_evidence(score=0.5)], top_score=0.5
    )
    result = await generate(gateway, settings, request)
    assert result.status == AnswerStatus.clarification
    assert result.clarification == "Which device are you using?"


async def test_validation_can_be_disabled():
    disabled = settings.model_copy(update={"validation_enabled": False})
    evidence = [make_evidence(score=0.9), make_evidence(score=0.85, ordinal=1)]
    gateway = FakeGateway("Paris is the capital [1].")
    request = GenerateRequest(question="capital?", evidence=evidence, top_score=0.9)
    result = await generate(gateway, disabled, request)
    assert result.status == AnswerStatus.answered
    assert len(gateway.chat_calls) == 1  # no verification call


async def test_refuses_an_answer_of_only_citation_markers():
    """A small model can emit bare markers; that must not pass as an answer."""
    request = GenerateRequest(question="q", evidence=[make_evidence(score=0.99)], top_score=0.99)
    result = await generate(FakeGateway("[4] [1] [2] [3]"), settings, request)
    assert result.status == AnswerStatus.refused
    assert result.refusal_reason == RefusalReason.unsupported
    # Nothing was drafted, so there is no confidence to report.
    assert result.confidence == 0.0


async def test_screenshot_text_is_shown_to_the_model():
    """A question like "why am I seeing this error?" is only answerable with it."""
    gateway = FakeGateway(replies=["Regenerate the key [1].", "SUPPORTED"])
    request = GenerateRequest(
        question="Why am I seeing this error?",
        evidence=[make_evidence(score=0.9, text="An invalid key error means the key is wrong.")],
        top_score=0.9,
        screenshot_text="Error: Invalid API key",
    )
    result = await generate(gateway, settings, request)
    assert result.status == AnswerStatus.answered
    prompt = gateway.chat_calls[0][1]["content"]
    assert "Invalid API key" in prompt
    assert prompt.index("Invalid API key") < prompt.index("Question:")


async def test_judge_alone_cannot_refuse_an_answer_the_passages_state():
    """llama3.2:3b judged this exact pair UNSUPPORTED at temperature 0."""
    evidence = [make_evidence(score=0.83, text="Monthly plans are non-refundable.")]
    gateway = FakeGateway(replies=["No, monthly plans are non-refundable [1].", "UNSUPPORTED"])
    request = GenerateRequest(question="Can I get a refund?", evidence=evidence, top_score=0.83)
    result = await generate(gateway, settings, request)
    assert result.status == AnswerStatus.answered
    assert result.support == "partial"


async def test_a_number_the_passages_do_not_state_is_refused():
    evidence = [make_evidence(score=0.9, text="The Free plan allows up to 3 members.")]
    gateway = FakeGateway(replies=["The Free plan allows 5 members [1].", "SUPPORTED"])
    request = GenerateRequest(question="How many members?", evidence=evidence, top_score=0.9)
    result = await generate(gateway, settings, request)
    assert result.status == AnswerStatus.refused
    assert result.refusal_reason == RefusalReason.unsupported


async def test_an_answer_without_citations_is_refused_not_credited_with_every_source():
    """Uncited drafts used to pass with every passage attached as a citation."""
    evidence = [make_evidence(score=0.9), make_evidence(score=0.85, ordinal=1)]
    gateway = FakeGateway("Paris is the capital of France.")
    request = GenerateRequest(question="capital?", evidence=evidence, top_score=0.9)
    result = await generate(gateway, settings, request)
    assert result.status == AnswerStatus.refused
    assert result.refusal_reason == RefusalReason.unsupported
    assert result.citations == []
    assert len(gateway.chat_calls) == 1  # not worth a judge call


async def test_the_judge_runs_on_the_judge_model_not_the_chat_model():
    """A second model cannot grade its own homework the way one model does."""
    evidence = [make_evidence(score=0.9)]
    gateway = FakeGateway(replies=["Paris is the capital [1].", "SUPPORTED"])
    request = GenerateRequest(question="capital?", evidence=evidence, top_score=0.9)
    await generate(gateway, settings, request)
    assert gateway.models_used == [None, settings.judge_model]


async def test_full_width_citation_markers_are_read_as_citations():
    """gpt-oss cites in the 【1】 style it was trained on; that is still a citation."""
    evidence = [make_evidence(score=0.9)]
    gateway = FakeGateway(replies=["The Team plan allows unlimited members. 【1】", "SUPPORTED"])
    request = GenerateRequest(question="members?", evidence=evidence, top_score=0.9)
    result = await generate(gateway, settings, request)
    assert result.status == AnswerStatus.answered
    assert result.answer == "The Team plan allows unlimited members. [1]"
    assert result.coverage == 1.0
