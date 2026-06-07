"""Configured store list helpers."""

from __future__ import annotations

from typing import Any

from src.config import settings
from src.providers import store_links


def list_available_stores() -> dict[str, Any]:
    """Return stores from config for the agent and UI."""
    stores = settings.stores_list
    details = []
    for name in stores:
        resolved = store_links.resolve_store(name)
        details.append({
            "name": resolved[0] if resolved else name,
            "domain": resolved[1] if resolved else "",
        })
    return {
        "stores": stores,
        "country": settings.default_country,
        "locale": settings.locale_hint,
        "currency": settings.currency,
        "location": settings.default_location or None,
        "details": details,
    }
