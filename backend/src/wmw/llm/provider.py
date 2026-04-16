"""LLM provider for word combination/subtraction."""

from typing import Protocol


class LLMProvider(Protocol):
    """Abstract interface for LLM inference.

    Swap models by implementing this protocol.
    """

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate a completion for the given prompt."""
        ...
