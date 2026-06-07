"""Search for parts, materials, and store availability."""

from __future__ import annotations

import re
from typing import Any

from src.agent.inventory import product_title_matches
from src.agent.tools.guides import extract_price_hint, extract_product_name
from src.config import settings
from src.providers import store_links, web_search


async def search_parts(
    part_name: str,
    project: str = "",
    specs: str = "",
    max_results: int = 8,
) -> dict[str, Any]:
    """Identify exact parts, specs, SKUs, and buying options."""
    locale = settings.locale_hint
    spec_bit = f" {specs}" if specs else ""
    project_ctx = f" for {project}" if project else ""
    queries = [
        f"{part_name}{spec_bit}{project_ctx} exact product specifications {locale}",
        f"{part_name}{spec_bit} SKU model number buy {locale}",
        f"what {part_name} do I need{project_ctx} {locale}",
    ]
    seen: set[str] = set()
    combined: list[dict[str, str]] = []
    for query in queries:
        for r in await web_search.search_web_async(query, max_results=max_results):
            url = r.get("url", "")
            if url not in seen:
                seen.add(url)
                combined.append(r)
    return {
        "part_name": part_name,
        "project": project,
        "specs": specs,
        "queries": queries,
        "results": combined[:12],
    }


def _product_page_score(url: str, domain: str, title: str, snippet: str) -> int:
    lower = url.lower()
    score = 0
    if domain and domain in lower:
        score += 20
    if re.search(r"[_/]p\d{4,}", lower):
        score += 25
    elif re.search(r"/dp/[A-Z0-9]", url):
        score += 25
    elif "/product" in lower and "?" not in lower:
        score += 10
    if re.search(r"/products/[^/]+/[^/]+/[^/]+$", lower) and "_p" not in lower:
        score -= 15
    if extract_price_hint(snippet, title) != "Check site":
        score += 5
    if "search" in lower and "?" in lower:
        score -= 8
    return score


async def search_part_at_store(
    part_name: str,
    store: str,
    specs: str = "",
    max_results: int = 6,
) -> dict[str, Any]:
    """Find the exact product at a store using multiple targeted searches."""
    location = settings.default_location.strip()
    locale = settings.locale_hint
    resolved = store_links.resolve_store(store)
    display = resolved[0] if resolved else store
    domain = resolved[1] if resolved else ""
    spec_bit = f" {specs}" if specs else ""

    queries = [
        store_links.build_site_search_query(store, f"{part_name}{spec_bit}"),
        f"site:{domain} {part_name}{spec_bit}" if domain else f"{part_name} {store}{spec_bit}",
        f'"{part_name}"{spec_bit} {display} buy price {locale}',
        f"{part_name}{spec_bit} {display} product",
    ]
    if location:
        queries = [f"{q} {location}" for q in queries[:2]] + queries[2:]

    seen_urls: set[str] = set()
    ranked: list[tuple[int, dict[str, str]]] = []

    for query in queries:
        for r in await web_search.search_web_async(query, max_results=max_results):
            url = r.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            score = _product_page_score(url, domain, r.get("title", ""), r.get("snippet", ""))
            ranked.append((score, r))

    ranked.sort(key=lambda x: x[0], reverse=True)
    search_url = store_links.build_store_search_url(store, f"{part_name}{spec_bit}")

    best: dict[str, str] | None = None
    product_url = ""
    for score, r in ranked:
        if score >= 25:
            picked, link_type = store_links.pick_store_url(store, part_name, r.get("url", ""))
            if link_type == "product":
                product_url = picked
                best = r
                break

    if not best and ranked:
        best = ranked[0][1]
        picked, link_type = store_links.pick_store_url(store, part_name, best.get("url", ""))
        if link_type == "product":
            product_url = picked

    title = best.get("title", "") if best else part_name
    snippet = best.get("snippet", "") if best else ""

    if best and not product_title_matches(part_name, title):
        best = None
        product_url = ""
        title = part_name
        snippet = ""

    exact_name = extract_product_name(title, display) if best else part_name
    if not product_title_matches(part_name, exact_name):
        exact_name = part_name
        product_url = ""

    price_hint = extract_price_hint(snippet, title) if snippet else "Search store"

    product_id = ""
    if product_url:
        m = re.search(r"[_/]p(\d+)", product_url, re.I)
        if m:
            product_id = m.group(1)

    store_options = [{
        "store": display,
        "exact_product_name": exact_name,
        "product_id": product_id,
        "price_hint": price_hint,
        "url": search_url,
        "product_url": product_url,
        "link_type": "product" if product_url else "search",
        "snippet": snippet or f"Exact match: {exact_name}" if product_url else f"Search {display} for '{part_name}'.",
        "confidence": "high" if product_url else ("medium" if ranked else "low"),
        "title": title,
        "alternatives": [
            {
                "title": extract_product_name(r.get("title", ""), display),
                "url": store_links.pick_store_url(store, part_name, r.get("url", ""))[0],
                "snippet": r.get("snippet", "")[:120],
            }
            for _, r in ranked[1:4]
        ],
    }]

    return {
        "part_name": part_name,
        "store": store,
        "specs": specs,
        "queries": queries,
        "search_url": search_url,
        "options": store_options,
        "match_quality": "exact" if product_url else ("possible" if ranked else "search_only"),
    }
