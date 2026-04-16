import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerProvider:
    """Embedding provider backed by sentence-transformers.

    Works with any model on HuggingFace that sentence-transformers supports,
    including Qwen3-Embedding-8B with Matryoshka dimension truncation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "cuda",
        truncate_dim: int | None = 512,
    ):
        self._model = SentenceTransformer(
            model_name,
            device=device,
            truncate_dim=truncate_dim,
        )
        self._dimension: int = truncate_dim or self._model.get_embedding_dimension() or 512

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Embed a batch of texts. Returns L2-normalized vectors."""
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns L2-normalized vector."""
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32)
