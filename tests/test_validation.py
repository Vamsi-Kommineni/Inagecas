from __future__ import annotations

from answer_service.validation import (
    TextCheck,
    Verdict,
    check_text,
    citation_coverage,
    compose_confidence,
    parse_verdict,
    settle_verdict,
    support_score,
)
from factories import make_evidence


def test_citation_coverage_full():
    assert citation_coverage("Paris is the capital [1].", [make_evidence()]) == 1.0


def test_citation_coverage_partial():
    answer = "Paris is the capital [1].\n\nIt is nice."
    assert citation_coverage(answer, [make_evidence()]) == 0.5


def test_a_citation_covers_the_rest_of_its_paragraph():
    """llama3.2:3b cites once up front; "According to [1], a. b. c." is fully cited."""
    answer = "According to [1], check spam first. Then request a new email. Filters delay mail."
    assert citation_coverage(answer, [make_evidence()]) == 1.0
    assert (
        citation_coverage("[1] Check spam first. Then request a new email.", [make_evidence()])
        == 1.0
    )


def test_a_paragraph_break_ends_what_a_citation_covers():
    answer = "Check spam first [1].\n\nAlso try turning it off and on."
    assert citation_coverage(answer, [make_evidence()]) == 0.5


def test_citation_coverage_none_without_markers():
    assert citation_coverage("Paris is the capital.", [make_evidence()]) == 0.0


def test_citation_coverage_ignores_out_of_range_markers():
    assert citation_coverage("See [5].", [make_evidence()]) == 0.0


def test_citation_coverage_rejects_an_answer_of_only_markers():
    """A marker-only answer cites everything and claims nothing."""
    evidence = [make_evidence() for _ in range(4)]
    assert citation_coverage("[4] [1] [2] [3]", evidence) == 0.0
    assert citation_coverage("[1]", evidence) == 0.0


def test_citation_coverage_ignores_marker_only_sentences():
    evidence = [make_evidence()]
    assert citation_coverage("The plan allows 3 members [1]. [1].", evidence) == 1.0


def test_parse_verdict_variants():
    assert parse_verdict("SUPPORTED") == Verdict.supported
    assert parse_verdict("the answer is UNSUPPORTED") == Verdict.unsupported
    assert parse_verdict("CONTRADICTED by passage 2") == Verdict.contradicted
    assert parse_verdict("PARTIAL") == Verdict.partial
    assert parse_verdict("gibberish") == Verdict.partial


def test_compose_confidence_rewards_support():
    low = compose_confidence(0.9, 2, 1.0, support_score(Verdict.unsupported))
    high = compose_confidence(0.9, 2, 1.0, support_score(Verdict.supported))
    assert high > low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_partial_support_answers_only_with_strong_retrieval_and_full_coverage():
    strong = compose_confidence(0.8, 2, 1.0, support_score(Verdict.partial))
    weak = compose_confidence(0.5, 1, 0.5, support_score(Verdict.partial))
    assert strong >= 0.5
    assert 0.35 <= weak < 0.5  # the clarification band


def test_compose_confidence_rewards_source_diversity():
    one_source = compose_confidence(0.6, 1, 0.5, 0.6)
    three_sources = compose_confidence(0.6, 3, 0.5, 0.6)
    assert three_sources > one_source


_REFUND = "Monthly plans are non-refundable. Annual plans may be refunded within 30 days."


def test_text_support_is_high_when_the_words_come_from_the_passage():
    evidence = [make_evidence(text=_REFUND)]
    assert check_text("No, monthly plans are non-refundable [1].", evidence).support >= 0.8


def test_text_support_is_low_for_words_the_passages_never_use():
    evidence = [make_evidence(text=_REFUND)]
    assert check_text("Contact your bank to dispute the charge [1].", evidence).support < 0.3


def test_a_number_the_sources_do_not_state_is_reported():
    evidence = [make_evidence(text="The Free plan allows up to 3 members.")]
    assert check_text("The Free plan allows 5 members [1].", evidence).foreign_numbers == ("5",)
    assert check_text("The Free plan allows 3 members [1].", evidence).foreign_numbers == ()


def test_numbers_from_the_question_are_allowed():
    evidence = [make_evidence(text="The Free plan allows up to 3 members.")]
    answer = "No, 5 people is more than the Free plan allows [1]."
    check = check_text(answer, evidence, context="Can 5 people share the Free plan?")
    assert check.foreign_numbers == ()


def test_settle_verdict_turns_a_disputed_unsupported_into_partial():
    assert settle_verdict(Verdict.unsupported, TextCheck(0.8, ())) == Verdict.partial
    assert settle_verdict(Verdict.unsupported, TextCheck(0.3, ())) == Verdict.unsupported
    assert settle_verdict(Verdict.supported, TextCheck(0.8, ())) == Verdict.supported
    assert settle_verdict(Verdict.contradicted, TextCheck(0.9, ())) == Verdict.contradicted


def test_settle_verdict_cannot_bless_a_foreign_number():
    assert settle_verdict(Verdict.supported, TextCheck(0.9, ("5",))) == Verdict.unsupported


def test_a_marker_after_the_full_stop_cites_the_sentence_before_it():
    evidence = [make_evidence(), make_evidence()]
    assert citation_coverage("Teams hold ten members. [1]", evidence) == 1.0
    assert citation_coverage("Teams hold ten members. [1] Seats cost $5.", evidence) == 1.0
    assert citation_coverage("Teams hold ten members. Seats cost $5. [2]", evidence) == 1.0


def test_a_marker_closing_a_paragraph_cites_the_whole_paragraph():
    evidence = [make_evidence(), make_evidence()]
    assert citation_coverage("Check spam. Then wait 15 minutes. Then retry [1].", evidence) == 1.0
    assert citation_coverage("Check spam. Then retry [1].\nUnrelated claim.", evidence) == 0.6667
    assert citation_coverage("Check spam. Then retry [9].", evidence) == 0.0


def test_dots_inside_words_do_not_end_a_sentence():
    evidence = [make_evidence()]
    assert citation_coverage("Mail comes from no-reply@acme.example [1].", evidence) == 1.0
    assert citation_coverage("Seats cost $5.99 each. Billing is monthly.", evidence) == 0.0
