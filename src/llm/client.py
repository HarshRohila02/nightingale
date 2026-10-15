"""The client the grounded explainer asks for text (task 3b): an interface, and Ollama behind it.

:class:`LLMClient` is all the explainer needs: a prompt in, text out, within a timeout. Anything
that satisfies it can stand in, which is how the tests run it: with a fake, never a network.

:class:`OllamaClient` speaks Ollama's documented HTTP API (``POST /api/generate``, docs at
github.com/ollama/ollama, ``docs/api.md``): ``model``, ``prompt``, ``system`` and
``"stream": false``, with ``options.temperature`` and ``options.num_predict`` for the length; the
reply's ``response`` holds the text and ``done`` says it is complete. It sends the request through
a *transport*, a function from (url, JSON payload, timeout) to the parsed reply, so the HTTP layer
can be replaced in tests; the default uses the standard library, and nothing is installed.

**Which model** is the owner's decision (D-5): the client is given ``llm.model`` from
configs/config.yaml and assumes nothing about it. **Running it** is the owner's too: Ollama puts a
model on the GPU, so on the laptop a run needs the owner's yes each time (D-9, docs/11 §3), and
nothing here starts, pulls or calls a model by itself. The default address is Ollama's own,
on this machine only.

Reference: docs/02-architecture.md §2 (src/llm), docs/17 T6.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_BASE_URL",
    "LLMClient",
    "LLMError",
    "OllamaClient",
    "Transport",
    "urllib_transport",
]

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
"""Ollama's documented default port, on this machine."""

Transport = Callable[[str, dict[str, Any], float], Any]
"""(url, JSON payload, timeout in seconds) -> the reply, parsed from JSON."""


class LLMError(RuntimeError):
    """The model could not give an answer: unreachable, refused, malformed or incomplete."""


@runtime_checkable
class LLMClient(Protocol):
    """Anything that turns a prompt into text within a timeout."""

    def generate(self, prompt: str, *, system: str, timeout: float) -> str:
        """The model's text for ``prompt``. Raises on failure, including a timeout."""
        ...


def urllib_transport(url: str, payload: dict[str, Any], timeout: float) -> Any:
    """POST ``payload`` as JSON to ``url`` and parse the JSON reply (the standard library only)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 - a set URL
        return json.loads(reply.read().decode("utf-8"))


class OllamaClient:
    """Ollama's ``/api/generate``, one complete answer per call (no streaming).

    Args:
        model: The model's name, ``llm.model`` (D-5 decides which).
        base_url: Where Ollama listens; its own default, on this machine.
        temperature: ``llm.temperature``; low, since this explains, not writes.
        max_tokens: ``llm.max_tokens``, sent as ``num_predict``.
        transport: How the request is sent; :func:`urllib_transport` unless a test passes another.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.1,
        max_tokens: int = 512,
        transport: Transport | None = None,
    ) -> None:
        if not model:
            raise ValueError("an Ollama client needs a model name (llm.model)")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.transport = transport or urllib_transport

    def payload(self, prompt: str, system: str) -> dict[str, Any]:
        """The request body, as Ollama's API documents it."""
        return {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }

    def generate(self, prompt: str, *, system: str, timeout: float) -> str:
        try:
            reply = self.transport(
                f"{self.base_url}/api/generate", self.payload(prompt, system), timeout
            )
        except TimeoutError:
            raise
        except (OSError, ValueError) as error:
            raise LLMError(f"Ollama did not answer: {error}") from error
        if not isinstance(reply, dict):
            raise LLMError("Ollama's reply is not a JSON object")
        if reply.get("error"):
            raise LLMError(f"Ollama refused: {reply['error']}")
        text = reply.get("response")
        if not isinstance(text, str) or reply.get("done") is not True:
            raise LLMError("Ollama's reply has no complete response")
        return text
