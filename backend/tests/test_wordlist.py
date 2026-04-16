"""Tests for English wordlist loading and noun filtering."""

from wmw.vocab.wordlist import _is_noun, load_english_words


def test_is_noun_accepts_nouns():
    for word in ["fire", "water", "king", "castle", "dragon", "planet"]:
        assert _is_noun(word), f"{word} should be a noun"


def test_is_noun_rejects_non_nouns():
    for word in ["quickly", "beautiful", "very", "happily", "extremely"]:
        assert not _is_noun(word), f"{word} should not be a noun"


def test_load_english_words_returns_nouns():
    words = load_english_words(min_zipf=4.0)  # high threshold for speed
    assert len(words) > 100
    assert all(w["category"] == "english" for w in words)
    # All entries should have a word key
    assert all("word" in w for w in words)
    # Spot check some expected nouns
    word_set = {w["word"] for w in words}
    for expected in ["water", "fire", "house", "world", "time"]:
        assert expected in word_set, f"{expected} should be in the word list"


def test_load_english_words_excludes_non_alpha():
    words = load_english_words(min_zipf=4.0)
    for w in words:
        assert w["word"].isalpha(), f"{w['word']} contains non-alpha chars"


def test_load_english_words_all_lowercase():
    words = load_english_words(min_zipf=4.0)
    for w in words:
        assert w["word"] == w["word"].lower(), f"{w['word']} is not lowercase"
