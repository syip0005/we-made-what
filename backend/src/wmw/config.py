from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Embedding model
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    embedding_device: str = "cuda"
    embedding_dim: int = 512  # Matryoshka truncation (Qwen3-Embedding supports 32-4096)

    # Vocabulary
    data_dir: Path = Path(__file__).parent / "data"
    vocab_index_path: Path = Path(__file__).parent / "data" / "vocab.index"
    vocab_meta_path: Path = Path(__file__).parent / "data" / "vocab.json"
    min_word_zipf: float = 2.0  # wordfreq zipf threshold (2.0 ≈ top ~50k words)

    # Game
    default_time_limit: int = 120
    seed_word_count: int = 5
    target_word_count: int = 10
    combine_rate_limit: float = 0.5  # min seconds between combines per player

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "WMW_"}
