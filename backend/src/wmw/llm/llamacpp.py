"""LLM provider backed by llama-cpp-python (GGUF models)."""

import logging

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LlamaCppProvider:
    """Local LLM inference via llama-cpp-python.

    Loads a GGUF model file and runs inference on GPU.
    Default: Gemma-4-E4B-Uncensored.
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        verbose: bool = False,
    ):
        logger.info(f"Loading model from {model_path}...")
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        logger.info("Model loaded.")

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate a completion for the given prompt."""
        output = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        content = output["choices"][0]["message"]["content"]  # ty: ignore[not-subscriptable]
        return content.strip() if content else ""
