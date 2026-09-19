from __future__ import annotations

from ingestion_worker.parsers import parse


def test_html_strips_tags_and_scripts():
    html = "<html><body><script>evil()</script><h1>Title</h1><p>Body text</p></body></html>"
    result = parse("html", html)
    assert "evil()" not in result
    assert "Title" in result
    assert "Body text" in result


def test_text_normalizes_whitespace():
    result = parse("text", "a\r\nb\t\t  c\n\n\n\nd")
    assert "\r" not in result
    assert "\t" not in result
    assert "\n\n\n" not in result


def test_bytes_input_is_decoded():
    assert parse("text", b"hello bytes") == "hello bytes"
