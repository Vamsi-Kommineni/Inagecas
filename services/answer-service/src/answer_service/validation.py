"""Answer validation: coverage, support/contradiction, confidence composition.

An answer is only delivered if it is actually backed by the evidence. Coverage,
lexical support and confidence composition are pure and deterministic; the
support/contradiction verdict uses the chat model as a grounding judge with a
defensive default, and the text check keeps that judge honest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.schemas import Evidence
from inagecas_shared.structured import parse_field
from inagecas_shared.text import cited_indices, has_prose, strip_citations

# Sentences end at punctuation followed by space or end of text, so the dots in
# "acme.example" and "$5.99" do not split a sentence in two.
_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")
_LEADING_MARKER = re.compile(r"\s*\[\d+\]")
_CLOSING_MARKER = re.compile(r"\[\d+\][\s.!?]*$")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORD_TEXT = (
    "a an the and or but if then of to in on at by for from with as is are was were be "
    "been being it its this that these those you your i we our they their not no can "
    "could should would may might will do does did have has had about into over after "
    "before which what when where who how why also than more most very just only some "
    "any all each per"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

# An UNSUPPORTED verdict on an answer whose words come from the passages at
# least this much is treated as the judge's doubt, not a veto.
LEXICAL_SUPPORT_MIN = 0.6


class Verdict(StrEnum):
    supported = "supported"
    partial = "partial"
    unsupported = "unsupported"
    contradicted = "contradicted"


_SUPPORT_SCORE = {
    Verdict.supported: 1.0,
    Verdict.partial: 0.4,
    Verdict.unsupported: 0.2,
    Verdict.contradicted: 0.0,
}

# Confidence weights (sum to 1.0): retrieval, source diversity, coverage, support.
# Tuned so a PARTIAL answer clears the threshold only with strong retrieval and
# full coverage, and otherwise lands in the clarification band.
_W_RETRIEVAL = 0.35
_W_DIVERSITY = 0.10
_W_COVERAGE = 0.20
_W_SUPPORT = 0.35

_VERIFY_SYSTEM = (
    "You check whether an ANSWER is supported by the provided PASSAGES.\n"
    "Verdicts:\n"
    "- SUPPORTED: every claim is backed by the passages\n"
    "- PARTIAL: some claims are backed, others are not\n"
    "- UNSUPPORTED: the answer is not backed by the passages\n"
    "- CONTRADICTED: the answer conflicts with the passages\n"
    'Reply with JSON: {"verdict": "<SUPPORTED|PARTIAL|UNSUPPORTED|CONTRADICTED>"}'
)
VERDICT_SCHEMA = {
    "title": "verdict",
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"],
        }
    },
    "required": ["verdict"],
    "additionalProperties": False,
}

_CLARIFY_SYSTEM = (
    "The documentation only partially answers the user's question. "
    "Ask ONE short clarifying question to narrow it down. Output only the question."
)


def citation_coverage(answer: str, evidence: list[Evidence]) -> float:
    """Fraction of the answer's prose sentences backed by a citation.

    A marker covers its own sentence and the uncited sentences that follow it
    in the same paragraph: "According to [1], a. b. c." cites all three. The
    mirror image holds: a marker that closes a paragraph, "a. b. c [1].",
    cites the sentences before it. A paragraph break ends both. A marker right
    after a full stop, "a. [1] b.", also cites the sentence it follows, since
    models that cite trailing put the marker after the stop. Sentences that
    are nothing but markers make no claim, so they are not counted, though
    the source they name carries on.
    """
    valid = set(range(1, len(evidence) + 1))
    covered = total = 0
    for paragraph in answer.split("\n"):
        carried = False
        flags: list[bool] = []
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            cited = bool(cited_indices(sentence) & valid)
            if cited and flags and _LEADING_MARKER.match(sentence):
                flags[-1] = True
            if not has_prose(sentence):
                carried = carried or cited
                continue
            carried = carried or cited
            flags.append(carried)
        closing = _CLOSING_MARKER.search(paragraph.rstrip())
        if closing and cited_indices(closing.group()) & valid:
            flags = [True] * len(flags)
        covered += sum(flags)
        total += len(flags)
    return round(covered / total, 4) if total else 0.0


def _content_words(text: str) -> set[str]:
    words = _WORD.findall(text.lower())
    return {w for w in words if w.isdigit() or (len(w) > 2 and w not in _STOPWORDS)}


@dataclass(frozen=True)
class TextCheck:
    """How much of the answer's wording the sources account for."""

    support: float  # share of the answer's content words found in the sources
    foreign_numbers: tuple[str, ...]  # numbers the sources never state


def check_text(answer: str, evidence: list[Evidence], *, context: str = "") -> TextCheck:
    """Compare the answer's words with the passages (plus the question and
    screenshot text, given as ``context``).

    Numbers are the facts support answers turn on -- member counts, day limits,
    prices -- so they are reported separately: one the sources never mention
    makes the answer unsupported whatever the rest of the words say.
    """
    allowed = _content_words("\n".join(item.text for item in evidence) + "\n" + context)
    words = _content_words(strip_citations(answer))
    foreign = tuple(sorted(w for w in words if w.isdigit() and w not in allowed))
    support = round(len(words & allowed) / len(words), 4) if words else 0.0
    return TextCheck(support, foreign)


def settle_verdict(judge: Verdict, text: TextCheck) -> Verdict:
    """Reconcile the judge with the text.

    A small judge model calls verbatim quotes of a passage UNSUPPORTED. When the
    answer's words demonstrably come from the passages, that verdict lowers the
    confidence as PARTIAL instead of refusing outright. The reverse holds too:
    a judge cannot bless an answer that states a number the sources do not.
    """
    if text.foreign_numbers:
        return Verdict.unsupported
    if judge == Verdict.unsupported and text.support >= LEXICAL_SUPPORT_MIN:
        return Verdict.partial
    return judge


def support_score(verdict: Verdict) -> float:
    return _SUPPORT_SCORE[verdict]


def compose_confidence(
    retrieval: float, source_count: int, coverage: float, support: float
) -> float:
    diversity = min(source_count, 3) / 3
    score = (
        _W_RETRIEVAL * retrieval
        + _W_DIVERSITY * diversity
        + _W_COVERAGE * coverage
        + _W_SUPPORT * support
    )
    return round(min(max(score, 0.0), 1.0), 4)


def parse_verdict(text: str) -> Verdict:
    upper = parse_field(text, "verdict").upper()
    # Order matters: "UNSUPPORTED" contains "SUPPORT", so check it first.
    for verdict, token in (
        (Verdict.contradicted, "CONTRADICT"),
        (Verdict.unsupported, "UNSUPPORT"),
        (Verdict.partial, "PARTIAL"),
        (Verdict.supported, "SUPPORT"),
    ):
        if token in upper:
            return verdict
    return Verdict.partial


async def verify_support(
    gateway: ModelGateway, answer: str, evidence: list[Evidence], *, model: str | None = None
) -> Verdict:
    passages = "\n\n".join(f"[{i}] {item.text}" for i, item in enumerate(evidence, start=1))
    output = await gateway.chat(
        [
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user", "content": f"PASSAGES:\n{passages}\n\nANSWER:\n{answer}\n\nVerdict:"},
        ],
        temperature=0.0,
        max_tokens=24,
        model=model,
        json_schema=VERDICT_SCHEMA,
    )
    return parse_verdict(output)


async def generate_clarification(gateway: ModelGateway, question: str) -> str:
    output = await gateway.chat(
        [
            {"role": "system", "content": _CLARIFY_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ],
        temperature=0.2,
        max_tokens=64,
    )
    return output.strip()
