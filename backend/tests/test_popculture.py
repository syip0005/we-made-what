"""Tests for pop culture normalization."""

from wmw.vocab.popculture import _normalize_entry


def test_normalize_lowercases():
    assert _normalize_entry("Darth Vader") == "darth vader"
    assert _normalize_entry("NARUTO") == "naruto"


def test_normalize_strips_whitespace():
    assert _normalize_entry("  goku  ") == "goku"


def test_normalize_rejects_empty():
    assert _normalize_entry("") is None
    assert _normalize_entry("   ") is None


def test_normalize_rejects_qids():
    assert _normalize_entry("Q12345") is None
    assert _normalize_entry("Q1") is None


def test_normalize_rejects_long_entries():
    assert _normalize_entry("a" * 61) is None


def test_normalize_accepts_qids_that_arent():
    # "Quantum" starts with Q but isn't a QID
    assert _normalize_entry("Quantum") == "quantum"
    assert _normalize_entry("Queen") == "queen"


def test_normalize_accepts_multi_word():
    assert _normalize_entry("Star Wars") == "star wars"
    assert _normalize_entry("Studio Ghibli") == "studio ghibli"
