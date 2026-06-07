"""Ollama URL and model name helpers."""

from __future__ import annotations


def normalize_ollama_base_url(url: str) -> str:
    """Ensure OpenAI-compatible base URL (…/v1), not native /api/chat."""
    url = url.rstrip("/")
    if url.endswith("/api/chat"):
        return url[: -len("/api/chat")] + "/v1"
    if not url.endswith("/v1"):
        return url + "/v1"
    return url


def resolve_ollama_model(requested: str, available: list[str]) -> str | None:
    """Match requested model to an installed name (e.g. llama3.1 → llama3.1:8b)."""
    if not available:
        return None
    if requested in available:
        return requested

    base = requested.split(":")[0]
    matches = [m for m in available if m.split(":")[0] == base]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    for preferred in (f"{base}:latest", f"{base}:8b"):
        if preferred in matches:
            return preferred
    return sorted(matches)[0]
