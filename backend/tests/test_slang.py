"""Tests for slang filtering logic."""

from wmw.vocab.slang import _is_useful_slang, _normalize_slang


def test_rejects_verb_descriptions():
    assert not _is_useful_slang("Slay", "To do something well.")
    assert not _is_useful_slang("Ghost", "to suddenly stop talking to someone")


def test_rejects_short_acronyms():
    assert not _is_useful_slang("TBH", "To be honest")
    assert not _is_useful_slang("NGL", "Not gonna lie")
    assert not _is_useful_slang("W", "Shorthand for win")
    assert not _is_useful_slang("L", "Shorthand for loss")
    assert not _is_useful_slang("IMO", "In my opinion")


def test_accepts_concept_slang():
    assert _is_useful_slang("Stan", "Supporting something obsessively")
    assert _is_useful_slang("Drip", "Another way of saying swag")
    assert _is_useful_slang("Bop", "An excellent song or album")
    assert _is_useful_slang("Fam", "A shorter word for family")


def test_accepts_longer_acronyms():
    # GOAT, FOMO etc. are >4 chars and have become real concepts
    assert _is_useful_slang("G.O.A.T", "The greatest of all time")
    assert _is_useful_slang("YOLO!", "You only live once")


def test_accepts_multi_word():
    assert _is_useful_slang("Cancel culture", "The practice of publicly rejecting someone")
    assert _is_useful_slang("Glow up", "A makeover or transformation")
    assert _is_useful_slang("Main character", "Someone who acts like the main character")


def test_rejects_single_char():
    assert not _is_useful_slang("W", "Shorthand for win")
    assert not _is_useful_slang("L", "Shorthand for loss")


def test_normalize_slang():
    assert _normalize_slang("Stan") == "stan"
    assert _normalize_slang("Glow Up") == "glow up"
    assert _normalize_slang("  Drip  ") == "drip"
    assert _normalize_slang("") is None
    assert _normalize_slang("a" * 61) is None  # too long


def test_normalize_preserves_multi_word():
    assert _normalize_slang("Cancel Culture") == "cancel culture"
    assert _normalize_slang("Main Character Energy") == "main character energy"
