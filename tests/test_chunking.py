from __future__ import annotations

from ingestion_worker.chunking import chunk_document


def test_markdown_splits_on_headings():
    text = "# Intro\nHello world.\n\n## Setup\nInstall the thing."
    chunks = chunk_document(text, "markdown")
    headings = [c.heading for c in chunks]
    assert "Intro" in headings
    assert "Setup" in headings


def test_ordinals_are_sequential():
    text = "# A\nsome text\n\n# B\nmore text"
    chunks = chunk_document(text, "markdown")
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_long_section_is_windowed_with_overlap():
    body = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_document(body, "text", max_words=40, overlap_words=10)
    assert len(chunks) > 1
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    # The window steps by (max - overlap), so the chunks share some words.
    assert set(first_words) & set(second_words)


def test_empty_document_yields_no_chunks():
    assert chunk_document("   \n  ", "markdown") == []


def test_non_markdown_is_single_section():
    chunks = chunk_document("plain text without headings", "text")
    assert len(chunks) == 1
    assert chunks[0].heading is None


def test_embedded_text_carries_title_and_heading():
    from inagecas_shared.text import with_context

    assert with_context("Troubleshooting", "Rate limit reached", "HTTP 429...") == (
        "Troubleshooting > Rate limit reached\nHTTP 429..."
    )
    assert with_context(None, None, "bare") == "bare"
    assert with_context("Doc", None, "text") == "Doc\ntext"
