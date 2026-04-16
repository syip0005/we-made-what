from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    """Abstract interface for embedding models.

    Swap models by implementing this protocol and registering in the registry.
    """

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape (len(texts), dimension)."""
        ...

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns shape (dimension,)."""
        ...
