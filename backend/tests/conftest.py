import numpy as np
import pytest

from wmw.vocab.index import VocabIndex


class FakeProvider:
    """Fake embedding provider for tests — uses random but deterministic vectors."""

    def __init__(self, dim: int = 32):
        self._dim = dim
        self._rng = np.random.default_rng(42)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_or_create(self, text: str) -> np.ndarray:
        if text not in self._cache:
            vec = self._rng.standard_normal(self._dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            self._cache[text] = vec
        return self._cache[text]

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        return np.stack([self._get_or_create(t) for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self._get_or_create(text)


@pytest.fixture
def fake_provider():
    return FakeProvider(dim=32)


@pytest.fixture
def sample_vocab(fake_provider):
    """Build a small VocabIndex with test words."""
    import faiss

    words = [
        "fire", "water", "steam", "ice", "earth",
        "king", "queen", "prince", "knight", "castle",
        "dog", "cat", "fish", "bird", "horse",
        "sword", "shield", "armor", "bow", "arrow",
        "naruto", "goku", "pikachu", "mario", "zelda",
    ]
    categories = [
        "english", "english", "english", "english", "english",
        "english", "english", "english", "english", "english",
        "english", "english", "english", "english", "english",
        "english", "english", "english", "english", "english",
        "anime", "anime", "game", "game", "game",
    ]

    embeddings = fake_provider.embed(words)

    index = faiss.IndexFlatIP(fake_provider.dimension)
    index.add(embeddings)

    return VocabIndex(words=words, categories=categories, index=index)
