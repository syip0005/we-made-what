import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerProvider:
    """Embedding provider backed by sentence-transformers.

    Works with any model on HuggingFace that sentence-transformers supports,
    including Qwen3-Embedding-8B with Matryoshka dimension truncation.

    Supports quantized loading via bitsandbytes (4-bit or 8-bit) to fit
    large models on smaller GPUs. E.g., 8B model at 4-bit uses ~4GB VRAM.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str | None = "cuda",
        truncate_dim: int | None = 512,
        quantization: str | None = None,
    ):
        model_kwargs: dict = {}

        if quantization == "4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
            )
            # bitsandbytes handles device placement
            device = None
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            device = None

        self._model = SentenceTransformer(
            model_name,
            device=device,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
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
