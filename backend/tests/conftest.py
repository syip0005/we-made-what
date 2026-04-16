import pytest


class FakeLLM:
    """Fake LLM provider for tests — returns deterministic results."""

    def __init__(self):
        self._responses: dict[str, str] = {}
        self._default = "steam"

    def set_response(self, prompt_contains: str, response: str):
        """Configure a response for prompts containing a given string."""
        self._responses[prompt_contains] = response

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        for key, response in self._responses.items():
            if key in prompt:
                return response
        return self._default


@pytest.fixture
def fake_llm():
    llm = FakeLLM()
    llm.set_response("fire", "steam")
    llm.set_response("king", "queen")
    llm.set_response("sword", "samurai")
    return llm
