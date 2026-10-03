"""Local LLM access through Ollama.

The LLM is used for *reasoning and language* only: planning, narrating results,
answering questions about the dataset, summarising feedback.  Every numeric
value in an answer comes from the deterministic artifacts produced by the ML
layer.

The client is deliberately defensive:

* the base URL and model come from configuration (never hard-coded),
* the health probe is cached so a missing Ollama does not slow the platform,
* every call has a timeout and returns a typed error instead of raising
  unexpected exceptions into the workflow.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import httpx

from config.logging_setup import get_logger
from config.settings import Settings, get_settings
from utils.errors import InvalidLLMResponseError, LLMUnavailableError

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """One completion plus its metadata."""

    text: str
    model: str
    elapsed_seconds: float = 0.0
    tokens: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    """Minimal, dependency-light Ollama REST client."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._health_cache: Dict[str, Any] = {"timestamp": 0.0, "available": False, "models": [], "error": None}

    # ------------------------------------------------------------------ urls
    @property
    def base_url(self) -> str:
        return str(self.settings.ollama_base_url).rstrip("/")

    @property
    def model(self) -> str:
        return str(self.settings.ollama_model)

    def _client(self, timeout: Optional[int] = None) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=timeout or self.settings.ollama_timeout,
            headers={"Content-Type": "application/json"},
        )

    # ---------------------------------------------------------------- health
    def health(self, refresh: bool = False) -> Dict[str, Any]:
        """Cached availability check (returns models, error message, latency)."""
        cache_seconds = max(int(self.settings.ollama_health_cache_seconds), 1)
        now = time.time()
        if not refresh and (now - float(self._health_cache["timestamp"])) < cache_seconds:
            return dict(self._health_cache)
        available = False
        models: List[str] = []
        error: Optional[str] = None
        started = time.perf_counter()
        try:
            with self._client(timeout=5) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
                payload = response.json()
                models = [str(item.get("name")) for item in payload.get("models", []) if item.get("name")]
                available = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]
            logger.info("Ollama is not reachable at %s (%s)", self.base_url, error)
        self._health_cache = {
            "timestamp": now,
            "available": available,
            "models": models,
            "error": error,
            "base_url": self.base_url,
            "configured_model": self.model,
            "model_installed": any(name.split(":")[0] == self.model.split(":")[0] for name in models) if models else None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return dict(self._health_cache)

    def is_available(self, refresh: bool = False) -> bool:
        if not self.settings.enable_llm:
            return False
        return bool(self.health(refresh=refresh).get("available"))

    def list_models(self, refresh: bool = False) -> List[str]:
        return list(self.health(refresh=refresh).get("models") or [])

    # ------------------------------------------------------------ generation
    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        format_json: bool = False,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> LLMResponse:
        """Single-turn completion."""
        if not self.settings.enable_llm:
            raise LLMUnavailableError(
                "LLM usage is disabled by configuration.",
                user_message="The language model is disabled (ENABLE_LLM=false).",
            )
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.settings.ollama_temperature if temperature is None else temperature,
                "num_ctx": int(self.settings.ollama_num_ctx),
                "num_predict": int(max_tokens or self.settings.ollama_num_predict),
            },
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"
        started = time.perf_counter()
        try:
            with self._client(timeout=timeout) as client:
                response = client.post("/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            message = exc.response.text[:200] if exc.response is not None else str(exc)
            raise LLMUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code if exc.response is not None else 'error'}: {message}",
                user_message=(
                    f"The model '{payload['model']}' could not be used. Run `ollama pull {payload['model']}` "
                    "or choose another model in Settings."
                ),
            ) from exc
        except Exception as exc:
            raise LLMUnavailableError(
                f"Ollama request failed: {exc}",
                user_message=(
                    f"Ollama is not reachable at {self.base_url}. Start it with `ollama serve` or update "
                    "OLLAMA_BASE_URL in Settings."
                ),
            ) from exc
        text = str(data.get("response", "")).strip()
        return LLMResponse(
            text=text,
            model=str(data.get("model", payload["model"])),
            elapsed_seconds=round(time.perf_counter() - started, 3),
            tokens=data.get("eval_count"),
            raw={"done_reason": data.get("done_reason"), "total_duration": data.get("total_duration")},
        )

    def chat(
        self,
        messages: Iterable[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        format_json: bool = False,
        timeout: Optional[int] = None,
    ) -> LLMResponse:
        """Multi-turn chat completion."""
        if not self.settings.enable_llm:
            raise LLMUnavailableError("LLM usage is disabled by configuration.")
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "options": {
                "temperature": self.settings.ollama_temperature if temperature is None else temperature,
                "num_ctx": int(self.settings.ollama_num_ctx),
                "num_predict": int(self.settings.ollama_num_predict),
            },
        }
        if format_json:
            payload["format"] = "json"
        started = time.perf_counter()
        try:
            with self._client(timeout=timeout) as client:
                response = client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            message = exc.response.text[:200] if exc.response is not None else str(exc)
            raise LLMUnavailableError(
                f"Ollama chat returned HTTP error: {message}",
                user_message=(
                    f"The configured model '{payload['model']}' is not usable. Run "
                    f"`ollama pull {payload['model']}` or pick another model in Settings."
                ),
            ) from exc
        except Exception as exc:
            raise LLMUnavailableError(
                f"Ollama chat request failed: {exc}",
                user_message=f"Ollama is not reachable at {self.base_url}.",
            ) from exc
        content = str((data.get("message") or {}).get("content", "")).strip()
        return LLMResponse(
            text=content,
            model=str(data.get("model", payload["model"])),
            elapsed_seconds=round(time.perf_counter() - started, 3),
            tokens=data.get("eval_count"),
        )

    def embeddings(self, text: str, *, model: Optional[str] = None) -> Optional[List[float]]:
        """Embedding vector (used by the optional RAG layer)."""
        try:
            with self._client(timeout=60) as client:
                response = client.post(
                    "/api/embeddings",
                    json={"model": model or self.model, "prompt": text},
                )
                response.raise_for_status()
                return list(response.json().get("embedding") or []) or None
        except Exception as exc:
            logger.debug("Embeddings unavailable: %s", exc)
            return None

    # ------------------------------------------------------------ structured
    def structured(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        retries: int = 2,
        default: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Ask for JSON and parse it defensively.

        The model is instructed to answer with a single JSON object; if parsing
        fails the prompt is retried once with an explicit repair instruction
        before falling back to ``default``.
        """
        attempt = 0
        last_error: Optional[str] = None
        working_prompt = prompt
        while attempt <= retries:
            try:
                response = self.generate(
                    working_prompt,
                    system=system,
                    format_json=True,
                    temperature=temperature,
                )
                parsed = extract_json(response.text)
                if parsed is not None:
                    return parsed
                last_error = "The response contained no parsable JSON object."
            except LLMUnavailableError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                last_error = f"{type(exc).__name__}: {exc}"
            attempt += 1
            working_prompt = (
                prompt
                + "\n\nIMPORTANT: reply with a single valid JSON object only - no prose, no markdown fences."
            )
        if default is not None:
            logger.warning("Falling back to the default structured answer (%s)", last_error)
            return default
        raise InvalidLLMResponseError(
            f"Could not parse a JSON answer from the model: {last_error}",
            user_message="The language model returned an answer that could not be parsed. Please try again.",
        )

    def safe_generate(self, prompt: str, *, fallback: str = "", **kwargs: Any) -> str:
        """Generate text, returning ``fallback`` if the LLM is unavailable."""
        try:
            return self.generate(prompt, **kwargs).text or fallback
        except (LLMUnavailableError, InvalidLLMResponseError) as exc:
            logger.info("LLM unavailable, using the deterministic fallback: %s", exc.user_message)
            return fallback

    def status(self) -> Dict[str, Any]:
        """Status payload for the API/UI."""
        health = self.health()
        return {
            "enabled": bool(self.settings.enable_llm),
            "available": bool(health.get("available")),
            "base_url": self.base_url,
            "configured_model": self.model,
            "models": health.get("models", []),
            "model_installed": health.get("model_installed"),
            "error": health.get("error"),
            "health_latency_ms": health.get("latency_ms"),
        }


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model response."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else {"value": payload}
    except json.JSONDecodeError:
        pass
    depth = 0
    start: Optional[int] = None
    for index, character in enumerate(cleaned):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = cleaned[start : index + 1]
                try:
                    payload = json.loads(candidate)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    start = None
    return None


_client: Optional[OllamaClient] = None


def get_llm_client(refresh: bool = False) -> OllamaClient:
    """Process-wide Ollama client."""
    global _client
    if _client is None or refresh:
        _client = OllamaClient()
    return _client


__all__ = ["LLMResponse", "OllamaClient", "extract_json", "get_llm_client"]
