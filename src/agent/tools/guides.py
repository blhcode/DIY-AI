"""Search for DIY guides and how-to content."""

from __future__ import annotations

import re
from typing import Any

from src.config import settings
from src.providers import web_search


async def search_diy_guides(project: str, max_results: int = 8) -> dict[str, Any]:
    """Find how-to guides, repair steps, and safety info for a project."""
    location = settings.default_location.strip()
    locale = settings.locale_hint
    loc_suffix = f" {location}" if location else f" {locale}"
    query = f"{project} DIY tutorial step by step how to{loc_suffix}"
    results = await web_search.search_web_async(query, max_results=max_results)
    return {
        "project": project,
        "query": query,
        "country": settings.default_country,
        "results": results,
        "safety_reminder": _safety_reminder(),
    }


async def search_detailed_instructions(
    project: str,
    max_results: int = 8,
) -> dict[str, Any]:
    """Deep search for comprehensive step-by-step instructions with specifics."""
    locale = settings.locale_hint
    queries = [
        f"{project} step by step beginner detailed measurements tools {locale}",
        f"{project} how to every step explained {locale}",
        f"how to {project} complete guide mm cm exact steps {locale}",
        f"{project} tutorial what to do first second third {locale}",
    ]
    seen_urls: set[str] = set()
    combined: list[dict[str, str]] = []
    for query in queries:
        results = await web_search.search_web_async(query, max_results=max_results)
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(r)
    return {
        "project": project,
        "queries": queries,
        "results": combined[: max_results * 2],
        "safety_reminder": _safety_reminder(),
    }


def _safety_reminder() -> str:
    safety = (
        "Always turn off power/water/gas before electrical, plumbing, or gas work. "
        "Hire a licensed professional for work beyond your skill level."
    )
    if settings.default_country == "AU":
        safety += (
            " In Australia, use a licensed electrician for most fixed wiring work "
            "and a licensed plumber for regulated plumbing/gas work."
        )
    return safety


def extract_price_hint(snippet: str, title: str = "") -> str:
    """Pull a price-like string from search snippet text."""
    text = f"{title} {snippet}"
    patterns = [
        r"\$\d+(?:\.\d{2})?(?:\s*[-–]\s*\$\d+(?:\.\d{2})?)?",
        r"£\d+(?:\.\d{2})?(?:\s*[-–]\s*£\d+(?:\.\d{2})?)?",
        r"€\d+(?:\.\d{2})?(?:\s*[-–]\s*€\d+(?:\.\d{2})?)?",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(0)
    return "Check site"


def extract_product_name(title: str, store_display: str) -> str:
    """Clean product title from search result."""
    name = title.strip()
    for suffix in (f" - {store_display}", f" | {store_display}", " - Bunnings Australia",
                   " - Mitre 10", " - Amazon.com.au", " - Amazon"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name or title
