from __future__ import annotations

from pathlib import Path

from evaluation.models import Ticket
from evaluation.runner import load_tickets
from evaluation.scoring import aggregate, normalize_expected, score_ticket, sources_covered

from factories import make_evidence
from inagecas_shared.schemas import AnswerStatus, AskResponse, Citation


def _citation(title: str, uri: str) -> Citation:
    evidence = make_evidence(title=title)
    return Citation(
        chunk_id=evidence.chunk_id,
        document_id=evidence.document_id,
        source_uri=uri,
        title=title,
        ordinal=0,
        score=0.8,
    )


def _response(
    status: AnswerStatus,
    citations: list[Citation] | None = None,
    answer: str | None = None,
) -> AskResponse:
    if status is AnswerStatus.answered and answer is None:
        answer = "Monthly plans are non-refundable [1]."
    return AskResponse(status=status, confidence=0.8, citations=citations or [], answer=answer)


def test_normalize_expected():
    assert normalize_expected("answered") == ["answered"]
    assert normalize_expected(["answered", "refused"]) == ["answered", "refused"]


def test_sources_covered_requires_all_expected():
    citations = [_citation("billing", "billing.md")]
    assert sources_covered(["billing"], citations) is True
    assert sources_covered(["troubleshooting"], citations) is False
    assert sources_covered([], citations) is True


def test_sources_covered_accepts_any_alternative():
    """The member limit is in billing and getting-started; either citation is right."""
    citations = [_citation("getting-started", "getting-started.md")]
    assert sources_covered([["billing", "getting-started"]], citations) is True
    assert sources_covered([["billing", "troubleshooting"]], citations) is False


def test_answered_with_correct_source_passes():
    ticket = Ticket(id="t", question="q", expect="answered", expect_sources=["billing"])
    outcome = score_ticket(
        ticket, _response(AnswerStatus.answered, [_citation("billing", "billing.md")])
    )
    assert outcome.passed
    assert outcome.retrieval_hit is True
    assert outcome.citations_ok is True


def test_answered_with_wrong_source_fails():
    ticket = Ticket(id="t", question="q", expect="answered", expect_sources=["troubleshooting"])
    outcome = score_ticket(
        ticket, _response(AnswerStatus.answered, [_citation("billing", "billing.md")])
    )
    assert not outcome.passed
    assert outcome.retrieval_hit is False


def test_answered_with_only_citation_markers_fails():
    """The pipeline can return bare markers; the harness must not score that a pass."""
    ticket = Ticket(id="t", question="q", expect="answered", expect_sources=["billing"])
    outcome = score_ticket(
        ticket,
        _response(AnswerStatus.answered, [_citation("billing", "billing.md")], answer="[1] [2]"),
    )
    assert not outcome.passed
    assert "no content" in (outcome.note or "")


def test_refused_answerable_ticket_counts_as_a_retrieval_miss():
    """Scoring it as unmeasurable would keep the miss out of retrieval recall."""
    ticket = Ticket(id="t", question="q", expect="answered", expect_sources=["billing"])
    outcome = score_ticket(ticket, _response(AnswerStatus.refused))
    assert outcome.retrieval_hit is False
    assert outcome.citations_ok is False


def test_wrong_status_fails():
    outcome = score_ticket(
        Ticket(id="t", question="q", expect="refused"), _response(AnswerStatus.answered)
    )
    assert not outcome.passed
    assert "expected" in (outcome.note or "")


def test_refused_when_expected_passes():
    outcome = score_ticket(
        Ticket(id="t", question="q", expect="refused"), _response(AnswerStatus.refused)
    )
    assert outcome.passed


def test_aggregate_computes_metrics_and_gaps():
    outcomes = [
        score_ticket(
            Ticket(
                id="a",
                category="how_to",
                question="q",
                expect="answered",
                expect_sources=["billing"],
            ),
            _response(AnswerStatus.answered, [_citation("billing", "billing.md")]),
        ),
        score_ticket(
            Ticket(
                id="b",
                category="how_to",
                question="q",
                expect="answered",
                expect_sources=["billing"],
            ),
            _response(AnswerStatus.refused),
        ),
        score_ticket(
            Ticket(id="c", category="off_topic", question="q", expect="refused"),
            _response(AnswerStatus.refused),
        ),
    ]
    report = aggregate(outcomes)
    assert report.total == 3
    assert report.passed == 2
    assert report.refusal_correct == 1.0
    assert report.answer_correct == 0.5
    assert report.retrieval_recall == 0.5  # the refused ticket counts as a miss
    assert "how_to/b" in report.doc_gaps
    assert report.by_category["off_topic"] == 1.0


def test_bundled_eval_set_loads():
    tickets = load_tickets(Path("evaluation/eval_sets/acme_support.yaml"))
    assert len(tickets) >= 8
    assert all(ticket.question for ticket in tickets)
