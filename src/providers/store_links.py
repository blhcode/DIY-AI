"""Retailer search URL builders and site: query helpers."""

from __future__ import annotations

from urllib.parse import quote_plus

# Store name (lowercase key) → (display name, domain for site: queries, search URL template)
STORE_REGISTRY: dict[str, tuple[str, str, str]] = {
    "home depot": (
        "Home Depot",
        "homedepot.com",
        "https://www.homedepot.com/s/{query}",
    ),
    "lowes": (
        "Lowe's",
        "lowes.com",
        "https://www.lowes.com/search?searchTerm={query}",
    ),
    "lowe's": (
        "Lowe's",
        "lowes.com",
        "https://www.lowes.com/search?searchTerm={query}",
    ),
    "menards": (
        "Menards",
        "menards.com",
        "https://www.menards.com/main/search.html?search={query}",
    ),
    "amazon": (
        "Amazon",
        "amazon.com",
        "https://www.amazon.com/s?k={query}",
    ),
    "harbor freight": (
        "Harbor Freight",
        "harborfreight.com",
        "https://www.harborfreight.com/catalogsearch/result?q={query}",
    ),
    "b&q": (
        "B&Q",
        "diy.com",
        "https://www.diy.com/search?term={query}",
    ),
    "bq": (
        "B&Q",
        "diy.com",
        "https://www.diy.com/search?term={query}",
    ),
    "screwfix": (
        "Screwfix",
        "screwfix.com",
        "https://www.screwfix.com/search?search={query}",
    ),
    "wickes": (
        "Wickes",
        "wickes.co.uk",
        "https://www.wickes.co.uk/search?q={query}",
    ),
    # Australia
    "bunnings": (
        "Bunnings",
        "bunnings.com.au",
        "https://www.bunnings.com.au/search/products?q={query}",
    ),
    "mitre 10": (
        "Mitre 10",
        "mitre10.com.au",
        "https://www.mitre10.com.au/search?q={query}",
    ),
    "mitre10": (
        "Mitre 10",
        "mitre10.com.au",
        "https://www.mitre10.com.au/search?q={query}",
    ),
    "amazon australia": (
        "Amazon Australia",
        "amazon.com.au",
        "https://www.amazon.com.au/s?k={query}",
    ),
    "amazon au": (
        "Amazon Australia",
        "amazon.com.au",
        "https://www.amazon.com.au/s?k={query}",
    ),
    "amazon.com.au": (
        "Amazon Australia",
        "amazon.com.au",
        "https://www.amazon.com.au/s?k={query}",
    ),
    "total tools": (
        "Total Tools",
        "totaltools.com.au",
        "https://www.totaltools.com.au/catalogsearch/result/?q={query}",
    ),
    "home timber & hardware": (
        "Home Timber & Hardware",
        "hth.com.au",
        "https://www.hth.com.au/search?q={query}",
    ),
    "home timber and hardware": (
        "Home Timber & Hardware",
        "hth.com.au",
        "https://www.hth.com.au/search?q={query}",
    ),
    "hth": (
        "Home Timber & Hardware",
        "hth.com.au",
        "https://www.hth.com.au/search?q={query}",
    ),
}


def resolve_store(store_name: str) -> tuple[str, str, str] | None:
    """Return (display_name, domain, search_url_template) for a store name."""
    key = store_name.strip().lower()
    if key in STORE_REGISTRY:
        return STORE_REGISTRY[key]
    for k, v in STORE_REGISTRY.items():
        if k in key or key in k:
            return v
    return None


def build_store_search_url(store_name: str, part_name: str) -> str:
    """Build a direct search URL on the retailer's site."""
    resolved = resolve_store(store_name)
    encoded = quote_plus(part_name)
    if resolved:
        _, _, template = resolved
        return template.format(query=encoded)
    return f"https://duckduckgo.com/?q={quote_plus(f'{part_name} {store_name} buy')}"


def build_site_search_query(store_name: str, part_name: str) -> str:
    """Build a DuckDuckGo site: query for a part at a store."""
    resolved = resolve_store(store_name)
    if resolved:
        _, domain, _ = resolved
        return f"site:{domain} {part_name}"
    return f'"{part_name}" {store_name} price buy'


def store_search_link(store_name: str, part_name: str) -> dict[str, str]:
    """Return store label and search URL for a part."""
    resolved = resolve_store(store_name)
    display = resolved[0] if resolved else store_name
    return {
        "store": display,
        "url": build_store_search_url(store_name, part_name),
    }


_BAD_URL_HOSTS = (
    "duckduckgo.com",
    "duck.co",
    "bing.com/aclick",
    "google.com/url",
    "facebook.com/l.php",
    "l.facebook.com",
)


def pick_store_url(store_name: str, part_name: str, candidate_url: str = "") -> tuple[str, str]:
    """
    Return (url, link_type) where link_type is 'product' or 'search'.
    Always returns a working retailer link — never a DDG redirect.
    """
    search_url = build_store_search_url(store_name, part_name)
    resolved = resolve_store(store_name)
    domain = resolved[1] if resolved else ""

    url = (candidate_url or "").strip()
    if not url.startswith("http"):
        return search_url, "search"

    lower = url.lower()
    if any(bad in lower for bad in _BAD_URL_HOSTS):
        return search_url, "search"

    if domain and domain in lower:
        # Product pages usually have longer paths; still valid retailer links
        if "/search" in lower or "searchTerm" in lower or "?q=" in lower or "?k=" in lower:
            return url, "search"
        return url, "product"

    return search_url, "search"
