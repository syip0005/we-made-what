"""Tests for WordMixer."""

from wmw.engine.mixer import WordMixer
from wmw.vocab.index import VocabIndex


def test_combine_returns_result(sample_vocab: VocabIndex):
    mixer = WordMixer(sample_vocab)
    result = mixer.combine("fire", "water")
    assert result is not None
    assert result.operation == "add"
    assert result.inputs == ("fire", "water")
    assert result.result not in ("fire", "water")  # inputs excluded
    assert 0 <= result.score <= 1.0
    assert isinstance(result.category, str)


def test_subtract_returns_result(sample_vocab: VocabIndex):
    mixer = WordMixer(sample_vocab)
    result = mixer.subtract("king", "queen")
    assert result is not None
    assert result.operation == "subtract"
    assert result.inputs == ("king", "queen")
    assert result.result not in ("king", "queen")


def test_combine_has_alternatives(sample_vocab: VocabIndex):
    mixer = WordMixer(sample_vocab)
    result = mixer.combine("fire", "water")
    assert result is not None
    assert len(result.alternatives) > 0
    for word, category, score in result.alternatives:
        assert isinstance(word, str)
        assert isinstance(category, str)
        assert isinstance(score, float)


def test_combine_unknown_word(sample_vocab: VocabIndex):
    mixer = WordMixer(sample_vocab)
    result = mixer.combine("fire", "nonexistent")
    assert result is None


def test_subtract_unknown_word(sample_vocab: VocabIndex):
    mixer = WordMixer(sample_vocab)
    result = mixer.subtract("nonexistent", "water")
    assert result is None


def test_combine_same_word(sample_vocab: VocabIndex):
    mixer = WordMixer(sample_vocab)
    result = mixer.combine("fire", "fire")
    assert result is not None
    # Result should not be "fire" (excluded)
    assert result.result != "fire"
