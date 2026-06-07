"""Free web search via DuckDuckGo — no API key required."""

from __future__ import annotations

from typing import Any

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None  # type: ignore[misc, assignment]


def search_web(
    query: str,
    max_results: int = 8,
    region: str = "wt-wt",
) -> list[dict[str, str]]:
    """Return title, url, snippet for a web search query."""
    if DDGS is None:
        return [{"title": "Search unavailable", "url": "", "snippet": "Install ddgs package."}]

    backends: list[str | None] = ["html", "lite", None]
    last_error = ""
    for backend in backends:
        try:
            with DDGS() as ddgs:
                kwargs: dict[str, Any] = {"region": region, "max_results": max_results}
                if backend:
                    kwargs["backend"] = backend
                results = list(ddgs.text(query, **kwargs))
            if results:
                return _normalize_results(results)
        except Exception as exc:
            last_error = str(exc)
            continue

    if last_error:
        return [{"title": "Search failed", "url": "", "snippet": last_error}]
    return []


def _normalize_results(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", r.get("link", "")),
            "snippet": r.get("body", r.get("snippet", "")),
        }
        for r in results
        if r.get("title") or r.get("href") or r.get("link")
    ]


async def search_web_async(
    query: str,
    max_results: int = 8,
    region: str | None = None,
) -> list[dict[str, str]]:
    import asyncio

    from src.config import settings

    reg = region or settings.ddg_region
    return await asyncio.to_thread(search_web, query, max_results, reg)
