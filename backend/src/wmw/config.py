from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM model (GGUF file path)
    model_path: str = ""  # path to .gguf file
    model_gpu_layers: int = -1  # -1 = all layers on GPU
    model_ctx_size: int = 2048

    # Game
    default_time_limit: int = 120
    seed_word_count: int = 5
    target_word_count: int = 10
    combine_rate_limit: float = 0.5  # min seconds between combines per player

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "WMW_"}
