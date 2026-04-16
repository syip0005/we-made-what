"""Core word-mixing engine: combine and subtract words via LLM."""

import logging
import re
from dataclasses import dataclass

from wmw.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

COMBINE_PROMPT = (
    "You are playing a word combination game. When given two words or concepts, "
    "you combine them into a single new word or short phrase (max 3 words) "
    "that creatively merges both ideas.\n\n"
    "Rules:\n"
    "- Reply with ONLY the result, nothing else\n"
    "- Keep it to 1-3 words maximum\n"
    "- Be creative but logical\n"
    "- Don't repeat either input word\n"
    "- The result should be a real thing, concept, or pop culture reference when possible\n\n"
    "Combine: {word_a} + {word_b}\n"
    "Result:"
)

SUBTRACT_PROMPT = (
    "You are playing a word subtraction game. When given two words or concepts, "
    "you subtract the second from the first - remove that quality/aspect "
    "and return what remains.\n\n"
    "Rules:\n"
    "- Reply with ONLY the result, nothing else\n"
    "- Keep it to 1-3 words maximum\n"
    "- Be creative but logical\n"
    "- Don't repeat either input word\n"
    "- The result should be a real thing, concept, or pop culture reference when possible\n\n"
    "Subtract: {word_a} - {word_b}\n"
    "Result:"
)


@dataclass
class MixResult:
    """Result of a word combination or subtraction."""

    inputs: tuple[str, str]
    operation: str  # "add" or "subtract"
    result: str


class WordMixer:
    """Core game logic: combine or subtract words via LLM generation.

    Results are cached so the same combination always gives the same answer.
    """

    def __init__(self, llm: LLMProvider):
        self._llm = llm
        self._cache: dict[tuple[str, str, str], str] = {}

    def combine(self, word_a: str, word_b: str) -> MixResult:
        """word_a + word_b -> LLM-generated result."""
        return self._mix(word_a, word_b, operation="add")

    def subtract(self, word_a: str, word_b: str) -> MixResult:
        """word_a - word_b -> LLM-generated result."""
        return self._mix(word_a, word_b, operation="subtract")

    def _mix(self, word_a: str, word_b: str, operation: str) -> MixResult:
        cache_key = (word_a.lower(), word_b.lower(), operation)

        if cache_key in self._cache:
            result = self._cache[cache_key]
        else:
            if operation == "add":
                prompt = COMBINE_PROMPT.format(word_a=word_a, word_b=word_b)
            else:
                prompt = SUBTRACT_PROMPT.format(word_a=word_a, word_b=word_b)

            raw = self._llm.generate(prompt, max_tokens=20)
            result = _clean_result(raw)
            self._cache[cache_key] = result
            logger.debug(f"{word_a} {'+' if operation == 'add' else '-'} {word_b} = {result}")

        return MixResult(
            inputs=(word_a, word_b),
            operation=operation,
            result=result,
        )

    @property
    def cache_size(self) -> int:
        return len(self._cache)


def _clean_result(raw: str) -> str:
    """Clean up LLM output to a usable word/phrase."""
    # Take only the first line
    result = raw.strip().split("\n")[0].strip()
    # Remove quotes, asterisks, markdown
    result = re.sub(r'[*"\'`]', "", result)
    # Remove any leading/trailing punctuation
    result = result.strip(".,!?;:-–—")
    # Lowercase
    result = result.lower().strip()
    # Truncate to max 3 words
    words = result.split()
    if len(words) > 3:
        result = " ".join(words[:3])
    return result or "nothing"
