"""Deterministic research — runs every time, does not rely on agent tool-calling."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx
from openai import AsyncOpenAI

from src.agent.inventory import filter_relevant_items, item_to_label, merge_items, normalize_item, normalize_tools_list
from src.agent.tools import guides, parts
from src.config import settings
from src.ollama_utils import normalize_ollama_base_url, resolve_ollama_model
from src.providers import web_search

logger = logging.getLogger(__name__)

MAX_STORE_LOOKUPS = 10  # max items to price-check at stores (materials only)
MAX_STORES_COMPARE = 2  # when user picks "all stores", search top 2 only (faster)

PARTS_EXTRACT_PROMPT = """You list materials and tools for ONE specific DIY/repair project only.
Given the project description and research snippets, output JSON only:

{
  "materials": [
    {"name": "clear searchable product name", "quantity": "e.g. 4 lengths", "specs": "exact size/type/grade"}
  ],
  "tools": [
    {"name": "reusable equipment only", "specs": "optional size"}
  ]
}

TOOLS vs MATERIALS — get this right:
- TOOLS = reusable equipment you already own or borrow: hammer, drill, saw, spanner, level, tape measure, screwdriver, clamps, ladder, safety gear
- MATERIALS = things you BUY and install/consume: timber, screws, nails, paint, pipe, brackets, sealant, glue, concrete, mailbox, house numbers
- NEVER put screws, nails, timber, paint, brackets, or pipe in the tools list
- ONLY items directly required for THIS project — nothing else
- NEVER include unrelated products (sports equipment, toys, food, electronics, clothing)
- Every material needs a clear "name" a shopper would search at a hardware store
- Include ALL lumber, hardware, fasteners, adhesives, sealants, fittings, consumables
- Include ALL tools the beginner must have before starting (with sizes)
- Use specs for dimensions/models; keep "name" short and searchable
- Region: LOCALE_PLACEHOLDER"""

FINALIZE_INVENTORY_PROMPT = """Build the complete shopping list for ONE DIY project.
Read the project, research, written steps, and any draft inventory. Output JSON only:

{
  "materials": [{"name": "searchable product name", "quantity": "...", "specs": "size/type"}],
  "tools": [{"name": "tool with size", "specs": ""}]
}

Rules:
- Include EVERY material/consumable/fastener used or implied in the steps
- Include EVERY reusable tool from tools_needed and per-step tool lists
- TOOLS = equipment (drill, saw, hammer). MATERIALS = buy/install items (timber, screws, paint)
- NEVER classify screws, nails, timber, brackets, paint, or pipe as tools
- Merge duplicates; use clear product names (e.g. "90×45mm H3 treated pine")
- ONLY items for THIS project — no unrelated products (sports, toys, food, etc.)
- Do NOT leave "name" empty — always a specific product or tool name
- Region: LOCALE_PLACEHOLDER"""

STEPS_EXTRACT_PROMPT = """You write DIY instructions for a complete beginner who has NEVER done this type of project.
Assume they do not know jargon. Explain every action as if teaching a friend with zero experience.
Output JSON only.

Schema:
{
  "tools_needed": ["reusable tools ONLY — hammer, drill, saw, spanner, level, tape measure, screwdriver"],
  "steps": [{
    "order": 1,
    "title": "short action title (verb first)",
    "instructions": "4-8 sentences of plain English. Say WHAT to do, HOW to hold/move tools, WHAT it looks like when correct.",
    "substeps": [
      "a. First micro-action with measurement or position",
      "b. Second micro-action — what you will see/hear/feel",
      "c. Third micro-action — how to check it is right"
    ],
    "tools": ["handheld tools for THIS step ONLY — not screws, timber, or paint"],
    "safety_note": "specific hazard for this step, or null"
  }],
  "difficulty": "easy|moderate|hard",
  "estimated_time": "realistic range",
  "estimated_cost": "AUD range using CURRENCY_PLACEHOLDER",
  "summary": "2-3 sentences"
}

MANDATORY RULES:
- 8-12 steps. EVERY step needs 2-4 substeps as STRINGS (lettered a, b, c… — not nested objects).
- Each instructions field: 3-5 clear sentences with measurements, tool sizes, and what success looks like.
- Describe physical cues: "the screw head should sit flush", "water stops dripping", "board sits level with pencil mark".
- Name parts precisely (e.g. "mixer cartridge", "M4 wood screw", "90×45mm pine stud") — never "the piece" or "the part".
- Region: LOCALE_PLACEHOLDER. Use local terms and CURRENCY_PLACEHOLDER.

FORBIDDEN — never write these (rewrite with specifics instead):
- "follow the manufacturer's instructions" / "refer to the manual"
- "if required" / "as needed" / "as necessary" / "when ready"
- "cut to size" without exact measurement and how to mark/cut
- "assemble according to instructions" — instead list each fastener, order, and orientation
- "use appropriate tools" — name the exact tool
- "attach securely" — say how many screws, where, and how tight

GOOD example for replacing a tap washer:
  title: "Turn off water and open the tap"
  instructions: "Locate the isolation valve under the sink — a small oval handle on the copper pipe. Turn it clockwise until it stops (usually a quarter turn). Turn the tap handle to the full-on hot position and leave it open for 10 seconds to release pressure. A trickle may come out then stop — that means the water is off."
  substeps: ["a. Kneel under the sink with a torch/phone light", "b. Turn the oval valve handle clockwise until resistance", "c. Open the tap fully and wait until dripping stops", "d. Place a towel in the basin to catch drips"]

BAD example (never output this):
  "Cut the box frame to size, if required. Assemble the pieces according to the manufacturer's instructions."

Use the research snippets for real techniques and measurements. Invent reasonable specifics when research is vague — beginners need numbers, not placeholders.
Output ONLY valid JSON."""

STEPS_QUICK_PROMPT = """Write clear DIY instructions for a complete beginner. Output JSON only.

{
  "tools_needed": ["reusable tools with sizes"],
  "steps": [{"order": 1, "title": "action title", "instructions": "2-4 sentences", "substeps": ["a. ...", "b. ..."], "tools": [], "safety_note": null}],
  "difficulty": "easy|moderate|hard",
  "estimated_time": "...",
  "estimated_cost": "... CURRENCY_PLACEHOLDER",
  "summary": "2 sentences"
}

- 8-10 steps, each with 2-3 substeps and exact actions (measurements, tool sizes)
- Region: LOCALE_PLACEHOLDER. Output ONLY valid JSON."""

STEPS_REFINE_PROMPT = """You rewrite vague DIY steps into beginner-proof instructions. Output JSON only with the same schema as before (tools_needed, steps with order/title/instructions/substeps/tools/safety_note, difficulty, estimated_time, estimated_cost, summary).

The user received steps that were too vague. Rewrite ALL steps from scratch using the research — do not keep generic phrases.

Rules:
- 12-16 steps, each with 3-6 substeps and 80+ words in instructions
- Replace every vague phrase with exact actions, measurements, tool sizes, and "what success looks like"
- FORBIDDEN: manufacturer's instructions, if required, as needed, cut to size (without mm/cm), assemble according to
- Region: LOCALE_PLACEHOLDER. Currency: CURRENCY_PLACEHOLDER
- Output ONLY valid JSON."""


async def verify_ollama() -> None:
    """Fail fast if Ollama is unreachable."""
    base_url = normalize_ollama_base_url(settings.ollama_base_url)
    root = base_url.rstrip("/")[:-3] if base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=8.0) as http:
        resp = await http.get(f"{root}/api/tags")
        resp.raise_for_status()


async def _ollama_client(timeout: float | None = None) -> tuple[AsyncOpenAI, str]:
    base_url = normalize_ollama_base_url(settings.ollama_base_url)
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="ollama",
        timeout=timeout or settings.ollama_timeout_seconds,
    )
    root = base_url.rstrip("/")[:-3] if base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=8.0) as http:
            tags = (await http.get(f"{root}/api/tags")).json()
        available = [m["name"] for m in tags.get("models", []) if m.get("name")]
    except Exception:
        available = []
    model = resolve_ollama_model(settings.ollama_model, available) or settings.ollama_model
    return client, model


def _trace_entry(tool: str, arguments: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(result, default=str)
    return {
        "tool": tool,
        "arguments": arguments,
        "result": raw,
        "result_preview": raw[:800],
    }


def _guide_text(guide_data: dict[str, Any], detailed_data: dict[str, Any]) -> str:
    lines: list[str] = []
    for data in (guide_data, detailed_data):
        for r in data.get("results") or []:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            if title or snippet:
                lines.append(f"{title}\n{snippet}")
    return "\n\n".join(lines[:16])


_VAGUE_PATTERNS = re.compile(
    r"|".join([
        r"manufacturer'?s?\s+instructions",
        r"\bif required\b",
        r"\bas needed\b",
        r"\bas necessary\b",
        r"follow (?:the )?instructions",
        r"refer to the (?:manual|guide)",
        r"cut (?:the )?\w+ to size\b",
        r"assemble (?:the )?(?:pieces|parts) according",
        r"use (?:the )?appropriate",
        r"\bwhen (?:ready|done|finished)\b",
        r"attach securely",
        r"as per (?:the )?instructions",
    ]),
    re.I,
)


def _normalize_substeps(raw: list[Any]) -> list[str]:
    out: list[str] = []
    for i, sub in enumerate(raw):
        letter = chr(ord("a") + i)
        if isinstance(sub, dict):
            title = str(sub.get("title", "")).strip()
            body = str(sub.get("instructions", sub.get("text", ""))).strip()
            if title and body:
                out.append(f"{letter}. {title}: {body}")
            elif body:
                out.append(f"{letter}. {body}")
            elif title:
                out.append(f"{letter}. {title}")
        else:
            text = str(sub).strip()
            if text and not re.match(r"^[a-z][.)]\s", text, re.I):
                text = f"{letter}. {text}"
            if text:
                out.append(text)
    return out


def _step_word_count(step: dict[str, Any]) -> int:
    n = len(str(step.get("instructions", "")).split())
    for sub in step.get("substeps") or []:
        if isinstance(sub, dict):
            n += len(str(sub.get("instructions", sub.get("text", ""))).split())
            n += len(str(sub.get("title", "")).split())
        else:
            n += len(str(sub).split())
    return n


def _normalize_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for s in steps:
        substeps = _normalize_substeps(s.get("substeps") or [])
        instructions = str(s.get("instructions", "")).strip()
        if _step_word_count({**s, "substeps": substeps}) < 50 and substeps:
            detail = " ".join(re.sub(r"^[a-z][.)]\s*", "", sub) for sub in substeps)
            if detail and detail not in instructions:
                instructions = f"{instructions} {detail}".strip()
        normalized.append({
            "order": int(s.get("order") or len(normalized) + 1),
            "title": str(s.get("title", f"Step {len(normalized) + 1}")).strip(),
            "instructions": instructions,
            "substeps": substeps,
            "tools": normalize_tools_list(s.get("tools") or []),
            "safety_note": s.get("safety_note"),
        })
    return normalized


def steps_are_detailed_enough(steps: list[dict[str, Any]]) -> bool:
    """True when steps have enough depth for a complete beginner."""
    if len(steps) < 10:
        return False
    word_counts = [_step_word_count(s) for s in steps]
    if sum(word_counts) / len(steps) < 55:
        return False
    vague = sum(
        1 for s in steps
        if _VAGUE_PATTERNS.search(str(s.get("instructions", "")))
        or _VAGUE_PATTERNS.search(" ".join(str(x) for x in (s.get("substeps") or [])))
    )
    if vague > max(2, len(steps) // 4):
        return False
    substeps = sum(len(s.get("substeps") or []) for s in steps)
    if substeps < len(steps) * 2:
        return False
    return True


async def _call_steps_llm(
    system_prompt: str,
    user: str,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    limit = timeout or settings.ollama_timeout_seconds
    client, model = await _ollama_client(timeout=limit)
    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        ),
        timeout=limit,
    )
    return json.loads(resp.choices[0].message.content or "{}")


async def _refine_steps_llm(
    query: str,
    research_text: str,
    parts_list: list[dict[str, str]],
    weak_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    parts_summary = "\n".join(
        f"- {p['name']} x{p.get('quantity', '1')}"
        + (f" ({p['specs']})" if p.get("specs") else "")
        for p in parts_list
    )
    prompt = (
        STEPS_REFINE_PROMPT.replace("CURRENCY_PLACEHOLDER", settings.currency)
        .replace("LOCALE_PLACEHOLDER", settings.locale_hint)
    )
    vague_sample = json.dumps(weak_steps[:4], indent=2)[:2000]
    user = (
        f"Project: {query}\n\n"
        f"These steps were REJECTED for being too vague:\n{vague_sample}\n\n"
        f"Parts:\n{parts_summary}\n\n"
        f"Research (use for real measurements and techniques):\n{research_text[:7000]}\n\n"
        "Rewrite the full step list with beginner-level detail."
    )
    try:
        return await _call_steps_llm(prompt, user)
    except Exception as exc:
        logger.warning("steps refine failed: %s", exc)
        return {}


def _stores_to_search(preferred_stores: list[str] | None) -> list[str]:
    if preferred_stores:
        return preferred_stores[:3]
    all_stores = settings.stores_list
    return all_stores[:MAX_STORES_COMPARE]


def _parse_inventory_json(data: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for m in data.get("materials") or data.get("parts") or []:
        norm = normalize_item(m, kind="material")
        if norm:
            items.append(norm)
    for t in data.get("tools") or []:
        norm = normalize_item(t, kind="tool")
        if norm:
            items.append(norm)
    return items


async def _extract_parts_llm(query: str, research_text: str) -> list[dict[str, str]]:
    prompt = PARTS_EXTRACT_PROMPT.replace("LOCALE_PLACEHOLDER", settings.locale_hint)
    user = (
        f"Project: {query}\n\nResearch:\n{research_text[:6000]}\n\n"
        "List all materials and tools for THIS project only."
    )
    try:
        client, model = await _ollama_client()
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return filter_relevant_items(_parse_inventory_json(data), query)
    except Exception as exc:
        logger.warning("parts LLM extract failed: %s", exc)
        return []


async def _finalize_inventory_llm(
    query: str,
    research_text: str,
    draft_items: list[dict[str, str]],
    step_data: dict[str, Any],
) -> list[dict[str, str]]:
    """Second pass: complete inventory from steps — catches items the first pass missed."""
    prompt = FINALIZE_INVENTORY_PROMPT.replace("LOCALE_PLACEHOLDER", settings.locale_hint)
    steps_json = json.dumps(step_data.get("steps") or [])[:6000]
    tools_needed = json.dumps(step_data.get("tools_needed") or [])
    draft = json.dumps(
        [{"name": p["name"], "quantity": p.get("quantity"), "specs": p.get("specs"), "kind": p.get("kind")}
         for p in draft_items],
        indent=2,
    )[:3000]
    user = (
        f"Project: {query}\n\n"
        f"Draft inventory:\n{draft}\n\n"
        f"tools_needed from steps: {tools_needed}\n\n"
        f"Steps:\n{steps_json}\n\n"
        f"Research:\n{research_text[:4000]}\n\n"
        "Output the COMPLETE final materials and tools list for this project."
    )
    try:
        data = await _call_steps_llm(prompt, user)
        finalized = _parse_inventory_json(data)
        merged = merge_items(draft_items, finalized)
        return filter_relevant_items(merged, query)
    except Exception as exc:
        logger.warning("inventory finalize failed: %s", exc)
        return filter_relevant_items(draft_items, query)


async def build_quick_steps(
    query: str,
    research_text: str,
    parts_list: list[dict[str, str]],
) -> dict[str, Any]:
    """Lighter step generation — used when the full pass fails or times out."""
    materials = [p for p in parts_list if p.get("kind") != "tool"]
    parts_summary = "\n".join(
        f"- {p['name']}" + (f" ({p['specs']})" if p.get("specs") else "") for p in materials[:8]
    )
    prompt = (
        STEPS_QUICK_PROMPT.replace("CURRENCY_PLACEHOLDER", settings.currency)
        .replace("LOCALE_PLACEHOLDER", settings.locale_hint)
    )
    user = (
        f"Project: {query}\n\nMaterials:\n{parts_summary or query}\n\n"
        f"Research:\n{research_text[:3500]}\n\nWrite 8-10 clear beginner steps."
    )
    try:
        step_data = await _call_steps_llm(prompt, user, timeout=settings.ollama_timeout_seconds)
        steps = _normalize_steps(step_data.get("steps") or [])
        step_data["steps"] = steps
        return step_data
    except Exception as exc:
        logger.warning("quick steps failed: %s", exc)
        return {}


async def build_detailed_steps(
    query: str,
    research_text: str,
    parts_list: list[dict[str, str]],
) -> dict[str, Any]:
    materials = [p for p in parts_list if p.get("kind") != "tool"]
    tools = [p for p in parts_list if p.get("kind") == "tool"]
    mat_lines = "\n".join(
        f"- {p['name']} x{p.get('quantity', '1')}" + (f" ({p['specs']})" if p.get("specs") else "")
        for p in materials
    )
    tool_lines = "\n".join(
        f"- {p['name']}" + (f" ({p['specs']})" if p.get("specs") else "") for p in tools
    )
    parts_summary = ""
    if mat_lines:
        parts_summary += f"Materials:\n{mat_lines}\n"
    if tool_lines:
        parts_summary += f"Tools:\n{tool_lines}\n"
    prompt = (
        STEPS_EXTRACT_PROMPT.replace("CURRENCY_PLACEHOLDER", settings.currency)
        .replace("LOCALE_PLACEHOLDER", settings.locale_hint)
    )
    user = (
        f"Project: {query}\n\n"
        f"Parts needed:\n{parts_summary or 'Identify from research'}\n\n"
        f"Research snippets (extract measurements, techniques, and order of work):\n"
        f"{research_text[:4000]}\n\n"
        "Write 8-12 beginner-proof steps. Every step needs substeps with exact actions."
    )
    try:
        step_data = await _call_steps_llm(
            prompt, user, timeout=settings.ollama_timeout_seconds
        )
        steps = _normalize_steps(step_data.get("steps") or [])
        if len(steps) >= 5:
            step_data["steps"] = steps
            return step_data
        logger.info("detailed steps returned only %d steps, trying quick pass", len(steps))
    except Exception as exc:
        logger.warning("steps LLM build failed: %s", exc)

    return await build_quick_steps(query, research_text, parts_list)


async def _store_lookup(
    pname: str,
    store: str,
    specs: str,
) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(
            parts.search_part_at_store(pname, store, specs=specs, max_results=4),
            timeout=25.0,
        )
    except Exception as exc:
        logger.warning("store lookup failed %s @ %s: %s", pname, store, exc)
        from src.providers import store_links

        resolved = store_links.resolve_store(store)
        display = resolved[0] if resolved else store
        return {
            "part_name": pname,
            "store": store,
            "search_url": store_links.build_store_search_url(store, pname),
            "options": [{
                "store": display,
                "exact_product_name": pname,
                "price_hint": "Check site",
                "url": store_links.build_store_search_url(store, pname),
                "product_url": "",
                "snippet": "Search timed out — use link below.",
                "confidence": "low",
            }],
        }


async def run_research_pipeline(
    query: str,
    preferred_stores: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Guides → parts → store lookups → step build → inventory finalize."""
    trace: list[dict[str, Any]] = []
    stores = _stores_to_search(preferred_stores)
    logger.info("pipeline start: query=%r stores=%s", query[:60], stores)

    guide_data, detailed_data = await asyncio.gather(
        guides.search_diy_guides(query, max_results=5),
        guides.search_detailed_instructions(query, max_results=5),
    )
    trace.append(_trace_entry("search_diy_guides", {"project": query}, guide_data))
    trace.append(_trace_entry("search_detailed_instructions", {"project": query}, detailed_data))

    research_text = _guide_text(guide_data, detailed_data)

    parts_list = await _extract_parts_llm(query, research_text)

    seen_names: set[str] = set()
    unique_parts: list[dict[str, str]] = []
    for p in parts_list:
        key = p["name"].lower()
        if key not in seen_names:
            seen_names.add(key)
            unique_parts.append(p)

    draft_parts = unique_parts or [{"name": query, "quantity": "1", "specs": "", "kind": "material"}]
    buy_items = [p for p in draft_parts if p.get("kind") != "tool"][:MAX_STORE_LOOKUPS]

    # Steps first — most important; don't wait for store price lookups
    step_data = await build_detailed_steps(query, research_text, draft_parts)

    lookup_tasks = [
        _store_lookup(p["name"], store, p.get("specs", ""))
        for p in buy_items
        for store in stores
    ]

    # Store lookups + inventory finalize in parallel (after steps are done)
    finalize_coro = _finalize_inventory_llm(query, research_text, draft_parts, step_data)
    if lookup_tasks:
        lookup_results, final_items = await asyncio.gather(
            asyncio.gather(*lookup_tasks),
            finalize_coro,
        )
    else:
        lookup_results = []
        final_items = await finalize_coro

    if not final_items:
        final_items = draft_parts

    idx = 0
    for part in buy_items:
        pname = part["name"]
        specs = part.get("specs", "")
        for store in stores:
            store_result = lookup_results[idx]
            idx += 1
            trace.append(
                _trace_entry(
                    "search_part_at_store",
                    {"part_name": pname, "store": store, "specs": specs},
                    store_result,
                )
            )

    trace.append(
        _trace_entry(
            "pipeline_meta",
            {"query": query},
            {"parts_identified": final_items, "stores_searched": stores},
        )
    )
    logger.info(
        "pipeline done: %d items (%d store lookups)",
        len(final_items),
        len(lookup_tasks),
    )
    return trace, step_data
