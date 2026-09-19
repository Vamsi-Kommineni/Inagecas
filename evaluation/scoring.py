"""Pure scoring: compare a ticket's expectations to the pipeline's response."""

from __future__ import annotations

from collections import defaultdict

from inagecas_shared.schemas import AskResponse, Citation
from inagecas_shared.text import has_prose

from .models import EvalReport, Ticket, TicketOutcome


def normalize_expected(expect: str | list[str]) -> list[str]:
    return [expect] if isinstance(expect, str) else list(expect)


def sources_covered(expected_sources: list[str | list[str]], citations: list[Citation]) -> bool:
    """True if every expected source appears in a citation.

    An entry is a substring, or a list of substrings of which any one will do:
    a fact stated in two documents can be cited from either.
    """
    haystacks = [f"{c.title or ''} {c.source_uri}".lower() for c in citations]

    def cited(source: str) -> bool:
        return any(source.lower() in hay for hay in haystacks)

    return all(
        any(cited(alt) for alt in ([entry] if isinstance(entry, str) else entry))
        for entry in expected_sources
    )


def score_ticket(ticket: Ticket, response: AskResponse) -> TicketOutcome:
    expected = normalize_expected(ticket.expect)
    actual = response.status.value
    status_ok = actual in expected

    # A ticket that should have been answered is scored for retrieval either way:
    # leaving it unmeasured would hide a retrieval miss from the recall figure.
    retrieval_hit: bool | None = None
    citations_ok: bool | None = None
    if ticket.expect_sources and (actual == "answered" or expected == ["answered"]):
        retrieval_hit = sources_covered(ticket.expect_sources, response.citations)
        citations_ok = bool(response.citations) and retrieval_hit

    # "Answered" with nothing but citation markers is not an answer.
    empty_answer = actual == "answered" and not has_prose(response.answer)

    passed = status_ok and retrieval_hit is not False and not empty_answer
    note: str | None = None
    if not status_ok:
        note = f"expected {expected}, got {actual}"
    elif empty_answer:
        note = "answered with citations but no content"
    elif retrieval_hit is False:
        note = "expected sources not cited"

    return TicketOutcome(
        id=ticket.id,
        category=ticket.category,
        expected=expected,
        actual=actual,
        passed=passed,
        retrieval_hit=retrieval_hit,
        citations_ok=citations_ok,
        note=note,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def aggregate(outcomes: list[TicketOutcome]) -> EvalReport:
    total = len(outcomes)
    passed = sum(o.passed for o in outcomes)

    should_refuse = [o for o in outcomes if o.expected == ["refused"]]
    should_answer = [o for o in outcomes if o.expected == ["answered"]]
    with_recall = [o for o in outcomes if o.retrieval_hit is not None]
    with_citations = [o for o in outcomes if o.citations_ok is not None]

    by_category: dict[str, list[bool]] = defaultdict(list)
    for outcome in outcomes:
        by_category[outcome.category].append(outcome.passed)

    # An answerable ticket the system could not answer points at a doc gap.
    doc_gaps = [
        f"{o.category}/{o.id}" for o in outcomes if "answered" in o.expected and not o.passed
    ]

    return EvalReport(
        total=total,
        passed=passed,
        pass_rate=_rate(passed, total) or 0.0,
        refusal_correct=_rate(
            sum(o.actual == "refused" for o in should_refuse), len(should_refuse)
        ),
        answer_correct=_rate(sum(o.passed for o in should_answer), len(should_answer)),
        retrieval_recall=_rate(sum(bool(o.retrieval_hit) for o in with_recall), len(with_recall)),
        citation_correct=_rate(
            sum(bool(o.citations_ok) for o in with_citations), len(with_citations)
        ),
        by_category={cat: _rate(sum(v), len(v)) or 0.0 for cat, v in by_category.items()},
        doc_gaps=doc_gaps,
        outcomes=outcomes,
    )
