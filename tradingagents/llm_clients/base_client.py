import logging
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_MAX_RATE_LIMIT_RETRIES = 5
_BASE_BACKOFF_SECONDS = 60


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True for any HTTP 429 from any provider SDK."""
    if getattr(exc, "status_code", None) == 429:
        return True
    response = getattr(exc, "response", None)
    return response is not None and getattr(response, "status_code", None) == 429


def _rate_limit_wait(exc: Exception) -> Optional[float]:
    """Extract seconds to wait from rate-limit error metadata, or return None."""
    # OpenRouter / OpenAI-compatible: reset timestamp buried in error body
    try:
        body = getattr(exc, "body", None) or {}
        headers = body.get("error", {}).get("metadata", {}).get("headers", {})
        reset_ms = headers.get("X-RateLimit-Reset")
        if reset_ms:
            return max(int(reset_ms) / 1000 - time.time(), 1.0)
    except Exception:
        pass
    # Standard Retry-After or X-RateLimit-Reset HTTP response headers
    try:
        resp_headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
        retry_after = resp_headers.get("Retry-After")
        if retry_after:
            return max(float(retry_after), 1.0)
        reset_ms = resp_headers.get("X-RateLimit-Reset")
        if reset_ms:
            return max(int(reset_ms) / 1000 - time.time(), 1.0)
    except Exception:
        pass
    return None


def invoke_with_retry(thunk: Callable) -> Any:
    """Call thunk(), retrying on rate-limit (429) errors with backoff.

    thunk must be a zero-argument callable that performs the LLM invoke.
    Closes over all call arguments via lambda at the call site.
    """
    for attempt in range(_MAX_RATE_LIMIT_RETRIES):
        try:
            return thunk()
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == _MAX_RATE_LIMIT_RETRIES - 1:
                raise
            wait = _rate_limit_wait(exc) or _BASE_BACKOFF_SECONDS * (2 ** attempt)
            msg = getattr(exc, "message", str(exc))
            logger.warning(
                "Rate limit hit: %s — retrying in %.0fs (attempt %d/%d)",
                msg, wait, attempt + 1, _MAX_RATE_LIMIT_RETRIES,
            )
            time.sleep(wait)


def normalize_content(response):
    """Normalize LLM response content to a plain string.

    Multiple providers (OpenAI Responses API, Google Gemini 3) return content
    as a list of typed blocks, e.g. [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}].
    Downstream agents expect response.content to be a string. This extracts
    and joins the text blocks, discarding reasoning/metadata blocks.
    """
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
    return response


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    def get_provider_name(self) -> str:
        """Return the provider name used in warning messages."""
        provider = getattr(self, "provider", None)
        if provider:
            return str(provider)
        return self.__class__.__name__.removesuffix("Client").lower()

    def warn_if_unknown_model(self) -> None:
        """Warn when the model is outside the known list for the provider."""
        if self.validate_model():
            return

        warnings.warn(
            (
                f"Model '{self.model}' is not in the known model list for "
                f"provider '{self.get_provider_name()}'. Continuing anyway."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the configured LLM instance."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Validate that the model is supported by this client."""
        pass
