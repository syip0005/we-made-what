"""Core word-mixing engine: combine and subtract words via embedding arithmetic."""

from dataclasses import dataclass

import numpy as np

from wmw.vocab.index import VocabIndex


@dataclass
class MixResult:
    """Result of a word combination or subtraction."""

    inputs: tuple[str, str]
    operation: str  # "add" or "subtract"
    result: str
    category: str
    score: float  # cosine similarity of result
    alternatives: list[tuple[str, str, float]]  # runner-ups: (word, category, score)


class WordMixer:
    """Core game logic: combine or subtract words via embedding arithmetic.

    Uses pre-computed vectors from the VocabIndex for speed —
    no model inference needed at game time.
    """

    def __init__(self, index: VocabIndex):
        self._index = index

    def combine(self, word_a: str, word_b: str) -> MixResult | None:
        """word_a + word_b -> nearest word in vocab.

        Returns None if either word is not in the vocabulary.
        """
        return self._mix(word_a, word_b, operation="add")

    def subtract(self, word_a: str, word_b: str) -> MixResult | None:
        """word_a - word_b -> nearest word in vocab.

        Returns None if either word is not in the vocabulary.
        """
        return self._mix(word_a, word_b, operation="subtract")

    def _mix(self, word_a: str, word_b: str, operation: str) -> MixResult | None:
        vec_a = self._index.get_vector(word_a)
        vec_b = self._index.get_vector(word_b)

        if vec_a is None or vec_b is None:
            return None

        if operation == "add":
            result_vec = vec_a + vec_b
        else:
            result_vec = vec_a - vec_b

        # L2-normalize the result before searching
        norm = np.linalg.norm(result_vec)
        if norm == 0:
            return None
        result_vec = result_vec / norm

        # Find nearest, excluding the input words
        candidates = self._index.nearest(
            result_vec, k=5, exclude={word_a, word_b}
        )

        if not candidates:
            return None

        top_word, top_cat, top_score = candidates[0]

        return MixResult(
            inputs=(word_a, word_b),
            operation=operation,
            result=top_word,
            category=top_cat,
            score=top_score,
            alternatives=candidates[1:],
        )
