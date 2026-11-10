"""LLM provider abstractions and implementations."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol


class ProviderError(RuntimeError):
    """A model request failed after the provider's retry policy."""


class LLMProvider(Protocol):
    """Minimal interface consumed by :class:`LLMMatcher`."""

    def complete(self, prompt: str, *, model: str, temperature: float | None) -> str:
        """Return the model's text response."""


class MockProvider:
    """Deterministic offline provider for examples and tests."""

    def __init__(
        self,
        responses: list[str] | None = None,
        responder: Callable[[str], str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responses = list(responses or ["MATCH"])
        self._responder = responder
        self._error = error
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, model: str, temperature: float | None) -> str:
        self.prompts.append(prompt)
        if self._error:
            raise self._error
        if self._responder:
            return self._responder(prompt)
        if not self._responses:
            raise ProviderError("Mock response queue is empty")
        return self._responses.pop(0)


class OpenAIProvider:
    """OpenAI Responses API provider with bounded transient retries.

    The client reads ``OPENAI_API_KEY`` from the environment by default. The
    import is lazy, so the rest of the package works without the optional SDK.
    """

    def __init__(self, *, timeout: float = 30.0, max_retries: int = 2) -> None:
        if timeout <= 0 or max_retries < 0:
            raise ValueError("timeout must be positive and max_retries non-negative")
        try:
            from openai import OpenAI
            from dotenv import load_dotenv
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Install the live provider with: pip install 'rodali-llm-attacker[openai]'"
            ) from exc
        # Load a local, ignored .env before the SDK resolves OPENAI_API_KEY.
        load_dotenv()
        # SDK retries are disabled so this class has one explicit retry policy.
        self._client = OpenAI(timeout=timeout, max_retries=0)
        self._max_retries = max_retries

    def complete(self, prompt: str, *, model: str, temperature: float | None) -> str:
        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The openai package is not installed") from exc

        transient = (APIConnectionError, APITimeoutError, RateLimitError)
        for attempt in range(self._max_retries + 1):
            try:
                request: dict[str, object] = {
                    "model": model,
                    "input": prompt,
                }
                # Omitting temperature is the explicit compatibility path for
                # models that do not expose that setting.
                if temperature is not None:
                    request["temperature"] = temperature
                response = self._client.responses.create(**request)
                return response.output_text
            except transient as exc:
                if attempt == self._max_retries:
                    raise ProviderError(
                        f"Transient model request failed after {attempt + 1} attempts: "
                        f"{type(exc).__name__}"
                    ) from exc
                time.sleep(0.5 * (2**attempt))
            except Exception as exc:
                raise ProviderError(f"Model request failed: {type(exc).__name__}") from exc
        raise AssertionError("unreachable")

