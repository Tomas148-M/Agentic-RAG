from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """A provider could not generate a valid response."""


class LLMProvider(ABC):
    """Common synchronous interface for text generation providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt (str): Input text supplied to the provider.

        Returns:
            str: Generated text.

        Raises:
            LLMProviderError: If generation fails or produces an invalid response.
        """
