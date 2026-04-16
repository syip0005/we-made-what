from wmw.embedding.provider import EmbeddingProvider
from wmw.embedding.sentence_tf import SentenceTransformerProvider


def create_provider(
    model_name: str,
    device: str = "cuda",
    truncate_dim: int | None = 512,
) -> EmbeddingProvider:
    """Create an embedding provider for the given model name.

    All sentence-transformers-compatible models go through
    SentenceTransformerProvider. Add branches here if you need
    a fundamentally different inference backend (e.g., llama-cpp, vLLM).
    """
    return SentenceTransformerProvider(
        model_name=model_name,
        device=device,
        truncate_dim=truncate_dim,
    )
