"""Tests for EUR-Lex text extraction.

The downloader writes a single ``text`` field (header + recitals + main_body)
while the manager historically expected separate structured fields.  These
tests pin the precedence so a record with a title never loses its body.
"""

from hmc.datasets.eurlex.manager import EURLexManager


def test_structured_fields_are_concatenated():
    rec = {"title": "T", "header": "H", "recitals": "R", "main_body": "B"}
    assert EURLexManager._extract_text(rec) == "T H R B"


def test_text_field_is_used_when_structured_fields_absent():
    rec = {"title": "T", "text": "H R B"}
    assert EURLexManager._extract_text(rec) == "T H R B"


def test_title_is_not_duplicated_when_text_already_starts_with_it():
    rec = {"title": "T", "text": "T H R"}
    assert EURLexManager._extract_text(rec) == "T H R"


def test_text_without_title():
    assert EURLexManager._extract_text({"text": "H R"}) == "H R"


def test_token_list_fallback():
    assert EURLexManager._extract_text({"token": ["a", "b"]}) == "a b"
