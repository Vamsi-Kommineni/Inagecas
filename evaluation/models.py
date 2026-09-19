"""Schemas for eval tickets and the resulting report."""

from __future__ import annotations

from pydantic import BaseModel, Field

from inagecas_shared.schemas import Turn


class Ticket(BaseModel):
    id: str
    category: str = "general"
    question: str
    # Earlier turns for a follow-up ticket, oldest first.
    history: list[Turn] = Field(default_factory=list)
    image: str | None = None
    # Acceptable outcome(s): "answered", "refused", or "clarification".
    expect: str | list[str]
    # Source titles/URIs (substring match) that good retrieval should surface.
    # An inner list means any one of those sources is acceptable.
    expect_sources: list[str | list[str]] = Field(default_factory=list)


class TicketOutcome(BaseModel):
    id: str
    category: str
    expected: list[str]
    actual: str
    passed: bool
    retrieval_hit: bool | None = None
    citations_ok: bool | None = None
    note: str | None = None


class EvalReport(BaseModel):
    total: int
    passed: int
    pass_rate: float
    refusal_correct: float | None = None
    answer_correct: float | None = None
    retrieval_recall: float | None = None
    citation_correct: float | None = None
    by_category: dict[str, float] = Field(default_factory=dict)
    doc_gaps: list[str] = Field(default_factory=list)
    outcomes: list[TicketOutcome] = Field(default_factory=list)
