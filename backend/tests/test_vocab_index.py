"""Tests for VocabIndex."""

import tempfile
from pathlib import Path

import numpy as np

from wmw.vocab.index import VocabIndex


def test_has_word(sample_vocab: VocabIndex):
    assert sample_vocab.has_word("fire")
    assert sample_vocab.has_word("naruto")
    assert not sample_vocab.has_word("nonexistent")


def test_get_category(sample_vocab: VocabIndex):
    assert sample_vocab.get_category("fire") == "english"
    assert sample_vocab.get_category("naruto") == "anime"
    assert sample_vocab.get_category("mario") == "game"
    assert sample_vocab.get_category("nonexistent") is None


def test_get_vector(sample_vocab: VocabIndex):
    vec = sample_vocab.get_vector("fire")
    assert vec is not None
    assert vec.shape == (32,)
    # Should be L2-normalized
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    assert sample_vocab.get_vector("nonexistent") is None


def test_nearest(sample_vocab: VocabIndex):
    vec = sample_vocab.get_vector("fire")
    results = sample_vocab.nearest(vec, k=3)
    assert len(results) == 3
    # First result should be "fire" itself (closest to its own vector)
    assert results[0][0] == "fire"
    # Each result is (word, category, score)
    for word, category, score in results:
        assert isinstance(word, str)
        assert isinstance(category, str)
        assert isinstance(score, float)


def test_nearest_with_exclude(sample_vocab: VocabIndex):
    vec = sample_vocab.get_vector("fire")
    results = sample_vocab.nearest(vec, k=3, exclude={"fire"})
    assert len(results) == 3
    assert all(word != "fire" for word, _, _ in results)


def test_random_words(sample_vocab: VocabIndex):
    words = sample_vocab.random_words(5)
    assert len(words) == 5
    assert all(sample_vocab.has_word(w) for w in words)


def test_random_words_with_category(sample_vocab: VocabIndex):
    words = sample_vocab.random_words(2, categories=["anime"])
    assert len(words) == 2
    assert all(sample_vocab.get_category(w) == "anime" for w in words)


def test_save_and_load(sample_vocab: VocabIndex):
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test.index"
        meta_path = Path(tmpdir) / "test.json"

        sample_vocab.save(index_path, meta_path)

        loaded = VocabIndex.load(index_path, meta_path)
        assert loaded.size == sample_vocab.size
        assert loaded.dimension == sample_vocab.dimension
        assert loaded.has_word("fire")
        assert loaded.get_category("naruto") == "anime"

        # Vectors should be identical after round-trip
        orig_vec = sample_vocab.get_vector("fire")
        loaded_vec = loaded.get_vector("fire")
        np.testing.assert_array_almost_equal(orig_vec, loaded_vec)


def test_size(sample_vocab: VocabIndex):
    assert sample_vocab.size == 25


def test_dimension(sample_vocab: VocabIndex):
    assert sample_vocab.dimension == 32
