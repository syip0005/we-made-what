"""FAISS-backed vocabulary index for nearest-neighbor word search."""

import json
from pathlib import Path

import faiss
import numpy as np


class VocabIndex:
    """Pre-computed embeddings for all vocabulary words, backed by FAISS.

    Uses IndexFlatIP (inner product) with L2-normalized vectors,
    which is equivalent to cosine similarity.
    """

    def __init__(
        self,
        words: list[str],
        categories: list[str],
        index: faiss.Index,
    ):
        self._words = words
        self._categories = categories
        self._word_to_idx: dict[str, int] = {w: i for i, w in enumerate(words)}
        self._index = index

    @property
    def size(self) -> int:
        return len(self._words)

    @property
    def dimension(self) -> int:
        return self._index.d

    def has_word(self, word: str) -> bool:
        return word in self._word_to_idx

    def get_category(self, word: str) -> str | None:
        idx = self._word_to_idx.get(word)
        if idx is None:
            return None
        return self._categories[idx]

    def get_vector(self, word: str) -> np.ndarray | None:
        """Get the pre-computed embedding for a known word."""
        idx = self._word_to_idx.get(word)
        if idx is None:
            return None
        return self._index.reconstruct(idx)

    def nearest(
        self,
        vector: np.ndarray,
        k: int = 5,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """Find k nearest words to the given vector.

        Returns [(word, category, score), ...] sorted by descending similarity.
        Over-fetches if excluding words, then filters.
        """
        exclude = exclude or set()
        # Over-fetch to account for filtered words
        fetch_k = k + len(exclude) + 5

        query = vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            word = self._words[idx]
            if word in exclude:
                continue
            results.append((word, self._categories[idx], float(score)))
            if len(results) >= k:
                break

        return results

    def random_words(
        self,
        n: int,
        categories: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[str]:
        """Pick n random words, optionally filtered by category."""
        rng = rng or np.random.default_rng()

        if categories:
            eligible = [
                i for i, cat in enumerate(self._categories) if cat in categories
            ]
        else:
            eligible = list(range(len(self._words)))

        chosen = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
        return [self._words[i] for i in chosen]

    def save(self, index_path: Path, meta_path: Path) -> None:
        """Persist index and metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        meta = {
            "words": self._words,
            "categories": self._categories,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False))

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "VocabIndex":
        """Load from disk (FAISS index + word metadata)."""
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text())
        return cls(
            words=meta["words"],
            categories=meta["categories"],
            index=index,
        )
