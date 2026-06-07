"""Map agent tool results into structured DIYPlanOut for the UI."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from openai import AsyncOpenAI

from src.agent.inventory import (
    classify_item_kind,
    filter_relevant_items,
    item_to_label,
    normalize_tools_list,
    product_title_matches,
)
from src.agent.research_pipeline import (
    build_detailed_steps,
    build_quick_steps,
    run_research_pipeline,
    verify_ollama,
)
from src.config import settings
from src.ollama_utils import normalize_ollama_base_url, resolve_ollama_model
from src.planner.models import DIYPlanOut, PartOut, StepOut, StoreOptionOut
from src.providers import store_links

SYNTHESIS_PROMPT = """You are a DIY plan formatter. Output a single JSON object from research + agent answer.

Schema:
{
  "mode": "plan",
  "title": "project title",
  "summary": "2-4 sentences",
  "difficulty": "easy|moderate|hard",
  "estimated_time": "e.g. 2-4 hours",
  "estimated_cost": "e.g. $45-90 AUD",
  "tools_needed": ["tool1", "tool2"],
  "steps": [{
    "order": 1,
    "title": "step name",
    "instructions": "4-8 sentences: precise actions, tools, measurements, what to check",
    "substeps": ["a. sub-action", "b. sub-action"],
    "tools": ["spanner"],
    "safety_note": "optional or null"
  }],
  "parts": [{
    "name": "part name",
    "quantity": "1",
    "notes": "exact specs: size, model, material",
    "store_options": [{
      "store": "Bunnings",
      "exact_product_name": "from tool",
      "product_id": "from tool",
      "price_hint": "$X or Check site",
      "url": "search_url from search_part_at_store tool",
      "product_url": "direct product page from tool",
      "link_type": "product|search",
      "snippet": "brief note",
      "confidence": "high|medium|low"
    }]
  }],
  "tips": ["tip"],
  "error": null
}

CRITICAL:
- The "parts" array is REQUIRED. Include EVERY part from search_part_at_store tool results.
- Each part MUST have store_options for each store searched (copy from tool JSON).
- MINIMUM 12 detailed steps with substeps where helpful.
- Never invent URLs — use search_url and product_url from tools only. No duckduckgo.com links.
- Output ONLY valid JSON.
"""

DIFFICULTY_LABELS = {"easy": "Easy", "moderate": "Moderate", "hard": "Hard"}
logger = logging.getLogger(__name__)

_SNIPPET_STEP_TITLE = re.compile(r"^step\s+\d+$", re.I)


def _steps_look_like_search_snippets(steps: list[dict[str, Any]]) -> bool:
    """Detect raw web snippets masquerading as steps (title 'Step 1', no substeps)."""
    if not steps:
        return False
    bad = 0
    for s in steps:
        title = str(s.get("title", "")).strip()
        substeps = s.get("substeps") or []
        if _SNIPPET_STEP_TITLE.match(title) and len(substeps) < 2:
            bad += 1
    return bad >= max(2, len(steps) // 2)


def _usable_steps(steps: list[dict[str, Any]]) -> bool:
    if not steps or _steps_look_like_search_snippets(steps):
        return False
    substantive = sum(
        1 for s in steps
        if len(str(s.get("instructions", ""))) > 40
        and not _SNIPPET_STEP_TITLE.match(str(s.get("title", "")).strip())
    )
    return substantive >= 4


def _steps_unavailable_message() -> dict[str, Any]:
    return {
        "order": 1,
        "title": "Instructions could not be generated",
        "instructions": (
            "The AI could not write step-by-step instructions this time — usually because "
            "Ollama timed out or was busy. Stop other Ollama tasks, pick one store in the "
            "dropdown, and try again with your project description."
        ),
        "substeps": [],
        "tools": [],
        "safety_note": None,
    }


def _tips_from_trace(tool_trace: list[dict[str, Any]]) -> list[str]:
    tips: list[str] = []
    seen: set[str] = set()
    for entry in tool_trace:
        if entry.get("tool") not in ("search_diy_guides", "search_detailed_instructions"):
            continue
        try:
            data = json.loads(entry.get("result") or "{}")
        except json.JSONDecodeError:
            continue
        tip = data.get("safety_reminder", "")
        if tip and tip not in seen:
            seen.add(tip)
            tips.append(tip)
    return tips


def _plan_shell(query: str, tool_trace: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": "plan",
        "title": query[:80] if query else "DIY Project",
        "summary": "",
        "difficulty": "moderate",
        "estimated_time": "Varies",
        "estimated_cost": "See parts list",
        "tools_needed": [],
        "steps": [],
        "parts": [],
        "tips": _tips_from_trace(tool_trace),
    }


async def synthesize_plan(
    agent_result: dict[str, Any],
    query: str,
    preferred_stores: list[str] | None = None,
) -> tuple[DIYPlanOut, str]:
    """Build DIYPlanOut from agent output; return (plan, summary)."""
    from src.agent.research_pipeline import verify_ollama

    await verify_ollama()

    agent_trace = agent_result.get("tool_trace") or []

    pipeline_trace, step_data = await run_research_pipeline(query, preferred_stores)
    tool_trace = pipeline_trace + agent_trace

    # Last resort: lighter prompt (faster, smaller JSON)
    if not _usable_steps(step_data.get("steps") or []):
        research = _research_text_from_trace(pipeline_trace)
        parts_raw = _parts_from_trace(pipeline_trace) or [
            {"name": query, "quantity": "1", "specs": "", "kind": "material"}
        ]
        logger.info("quick step retry for %r", query[:60])
        step_data = await build_quick_steps(query, research, parts_raw)

    plan_data = _plan_shell(query, tool_trace)
    plan_data = _apply_step_data(plan_data, step_data)
    plan_data = _merge_parts_from_trace(plan_data, tool_trace)
    plan_data = _merge_complete_inventory(
        plan_data, step_data, tool_trace, preferred_stores, query
    )
    plan_data.setdefault("title", query[:80])

    if not _usable_steps(plan_data.get("steps") or []):
        plan_data["steps"] = [_steps_unavailable_message()]

    plan = _dict_to_plan(plan_data, tool_trace)
    summary = (
        step_data.get("summary")
        or plan_data.get("summary")
        or _default_summary(plan)
    )
    return plan, summary


def _research_text_from_trace(tool_trace: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in tool_trace:
        if entry.get("tool") not in ("search_diy_guides", "search_detailed_instructions", "search_web"):
            continue
        try:
            data = json.loads(entry.get("result") or "{}")
        except json.JSONDecodeError:
            continue
        for r in data.get("results") or []:
            lines.append(f"{r.get('title', '')}\n{r.get('snippet', '')}")
    return "\n\n".join(lines)


def _parts_from_trace(tool_trace: list[dict[str, Any]]) -> list[dict[str, str]]:
    for entry in tool_trace:
        if entry.get("tool") != "pipeline_meta":
            continue
        try:
            data = json.loads(entry.get("result") or "{}")
        except json.JSONDecodeError:
            continue
        return data.get("parts_identified") or []
    return []


def _apply_step_data(plan_data: dict[str, Any], step_data: dict[str, Any]) -> dict[str, Any]:
    """Apply LLM-generated steps — never keep raw search snippets."""
    if not step_data:
        return plan_data
    steps = step_data.get("steps") or []
    if steps and not _steps_look_like_search_snippets(steps):
        plan_data["steps"] = steps
    if step_data.get("tools_needed"):
        plan_data["tools_needed"] = normalize_tools_list(step_data["tools_needed"])
    if step_data.get("difficulty"):
        plan_data["difficulty"] = step_data["difficulty"]
    if step_data.get("estimated_time"):
        plan_data["estimated_time"] = step_data["estimated_time"]
    if step_data.get("estimated_cost"):
        plan_data["estimated_cost"] = step_data["estimated_cost"]
    if step_data.get("summary"):
        plan_data["summary"] = step_data["summary"]
    return plan_data


def _build_context(
    query: str,
    tool_trace: list[dict[str, Any]],
    assistant_content: str,
    preferred_stores: list[str] | None,
) -> str:
    store_list = preferred_stores or settings.stores_list
    parts = [
        f"User query: {query}",
        f"Region: {settings.locale_hint} ({settings.default_country})",
        f"Currency: {settings.currency}",
        f"Stores: {', '.join(store_list)}",
    ]
    if preferred_stores:
        parts.append(f"User shops ONLY at: {', '.join(preferred_stores)}")
    if assistant_content:
        parts.append(f"Agent detailed answer (primary source for steps):\n{assistant_content}")
    for entry in tool_trace:
        parts.append(
            f"Tool: {entry['tool']}\nArguments: {json.dumps(entry.get('arguments', {}))}\n"
            f"Result: {entry.get('result', entry.get('result_preview', ''))}"
        )
    return "\n\n---\n\n".join(parts)


def _store_option_from_tool(
    opt: dict[str, Any],
    part_name: str,
    search_url: str = "",
) -> dict[str, Any]:
    store = opt.get("store", "")
    product_url = opt.get("product_url", "")
    search = search_url or store_links.build_store_search_url(store, part_name)
    exact = opt.get("exact_product_name", "") or part_name
    if not product_title_matches(part_name, exact):
        exact = part_name
        product_url = ""
    prod = ""
    if product_url:
        prod, ptype = store_links.pick_store_url(store, part_name, product_url)
        if ptype != "product":
            prod = ""
    return {
        "store": store,
        "exact_product_name": exact,
        "product_id": opt.get("product_id", ""),
        "price_hint": opt.get("price_hint", "Check site"),
        "url": search,
        "product_url": prod,
        "link_type": "product" if prod else "search",
        "snippet": opt.get("snippet", ""),
        "confidence": opt.get("confidence", "medium"),
    }


def _item_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _generic_store_option(store: str, item_name: str) -> dict[str, Any]:
    resolved = store_links.resolve_store(store)
    display = resolved[0] if resolved else store
    url = store_links.build_store_search_url(store, item_name)
    return {
        "store": display,
        "exact_product_name": item_name,
        "product_id": "",
        "price_hint": "Search store",
        "url": url,
        "product_url": "",
        "link_type": "search",
        "snippet": "",
        "confidence": "low",
    }


def _merge_complete_inventory(
    plan_data: dict[str, Any],
    step_data: dict[str, Any],
    tool_trace: list[dict[str, Any]],
    preferred_stores: list[str] | None,
    query: str = "",
) -> dict[str, Any]:
    """Ensure ALL materials and tools appear in parts list (from extraction + steps)."""
    parts_map: dict[str, dict[str, Any]] = {}
    for p in plan_data.get("parts") or []:
        name = p.get("name", "").strip()
        if name:
            parts_map[_item_key(name)] = {
                "name": name,
                "quantity": p.get("quantity", "1"),
                "notes": p.get("notes", ""),
                "category": p.get("category", "material"),
                "store_options": list(p.get("store_options") or []),
            }

    def add_item(
        name: str,
        *,
        quantity: str = "1",
        notes: str = "",
        category: str = "material",
    ) -> None:
        name = item_to_label(name) if not isinstance(name, str) else name.strip()
        if not name or len(name) < 2:
            return
        key = _item_key(name)
        resolved = classify_item_kind(name, default=category)
        if key in parts_map:
            existing = parts_map[key]
            if notes and not existing.get("notes"):
                existing["notes"] = notes
            # Materials trump tools when reclassifying (screws aren't tools)
            if resolved == "material" or existing.get("category") == "material":
                existing["category"] = "material"
            else:
                existing["category"] = resolved
            return
        parts_map[key] = {
            "name": name,
            "quantity": quantity,
            "notes": notes,
            "category": resolved,
            "store_options": [],
        }

    for entry in tool_trace:
        if entry.get("tool") != "pipeline_meta":
            continue
        try:
            data = json.loads(entry.get("result") or "{}")
        except json.JSONDecodeError:
            continue
        for p in data.get("parts_identified") or []:
            kind = p.get("kind", "material")
            category = "tool" if kind == "tool" else "material"
            add_item(
                p.get("name", ""),
                quantity=p.get("quantity", "1"),
                notes=p.get("specs", ""),
                category=category,
            )

    for tool in normalize_tools_list(step_data.get("tools_needed") or []):
        kind = classify_item_kind(tool, default="tool")
        add_item(tool, category=kind, notes="Required before you start")

    for step in plan_data.get("steps") or step_data.get("steps") or []:
        step_title = step.get("title", "")
        for tool in normalize_tools_list(step.get("tools") or []):
            kind = classify_item_kind(tool, default="tool")
            note = f"Used in step {step.get('order', '?')}: {step_title}" if step_title else "Required"
            add_item(tool, category=kind, notes=note)

    # Final reclassify pass
    for item in parts_map.values():
        item["category"] = classify_item_kind(
            item["name"], default=item.get("category", "material")
        )

    stores = preferred_stores or settings.stores_list[:2]
    for item in parts_map.values():
        if item.get("store_options"):
            continue
        if item.get("category") == "tool" and stores:
            item["store_options"] = [_generic_store_option(stores[0], item["name"])]
        elif item.get("category") != "tool" and stores:
            item["store_options"] = [_generic_store_option(stores[0], item["name"])]

    materials = [p for p in parts_map.values() if p.get("category") != "tool"]
    tools = [p for p in parts_map.values() if p.get("category") == "tool"]
    if query:
        materials = [
            p for p in materials
            if filter_relevant_items([{"name": p["name"], "kind": "material"}], query)
        ]
    materials.sort(key=lambda p: p["name"].lower())
    tools.sort(key=lambda p: p["name"].lower())
    plan_data["parts"] = materials + tools
    plan_data["tools_needed"] = [p["name"] for p in tools if p.get("name")]
    return plan_data


def _merge_parts_from_trace(
    plan_data: dict[str, Any],
    tool_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Always rebuild parts from tool trace so the parts list is never empty when searches ran."""
    parts_map: dict[str, dict[str, Any]] = {}

    for p in plan_data.get("parts") or []:
        name = p.get("name", "").strip()
        if name:
            parts_map[_item_key(name)] = {
                "name": name,
                "quantity": p.get("quantity", "1"),
                "notes": p.get("notes", ""),
                "category": p.get("category", "material"),
                "store_options": list(p.get("store_options") or []),
            }

    for entry in tool_trace:
        tool = entry.get("tool", "")
        args = entry.get("arguments") or {}
        try:
            data = json.loads(entry.get("result") or "{}")
        except json.JSONDecodeError:
            continue

        if tool == "pipeline_meta":
            for p in data.get("parts_identified") or []:
                name = p.get("name", "").strip()
                if not name:
                    continue
                key = _item_key(name)
                kind = p.get("kind", "material")
                category = "tool" if kind == "tool" else "material"
                if key not in parts_map:
                    parts_map[key] = {
                        "name": name,
                        "quantity": p.get("quantity", "1"),
                        "notes": p.get("specs", ""),
                        "category": category,
                        "store_options": [],
                    }
                elif p.get("specs") and not parts_map[key].get("notes"):
                    parts_map[key]["notes"] = p["specs"]

        elif tool == "search_part_at_store":
            part_name = (data.get("part_name") or args.get("part_name", "")).strip()
            if not part_name:
                continue
            specs = data.get("specs") or args.get("specs", "")
            key = _item_key(part_name)
            if key not in parts_map:
                parts_map[key] = {
                    "name": part_name,
                    "quantity": "1",
                    "notes": specs,
                    "category": "material",
                    "store_options": [],
                }
            elif specs and not parts_map[key].get("notes"):
                parts_map[key]["notes"] = specs

            existing_stores = {o.get("store") for o in parts_map[key]["store_options"]}
            for opt in data.get("options") or []:
                store = opt.get("store", "")
                if store in existing_stores:
                    continue
                existing_stores.add(store)
                parts_map[key]["store_options"].append(
                    _store_option_from_tool(opt, part_name, data.get("search_url", ""))
                )

        elif tool == "search_parts":
            part_name = (data.get("part_name") or args.get("part_name", "")).strip()
            specs = data.get("specs") or args.get("specs", "")
            if not part_name:
                continue
            key = _item_key(part_name)
            if key not in parts_map:
                note = specs
                results = data.get("results") or []
                if not note and results:
                    note = results[0].get("snippet", "")[:150]
                parts_map[key] = {
                    "name": part_name,
                    "quantity": "1",
                    "notes": note,
                    "category": "material",
                    "store_options": [],
                }

    plan_data["parts"] = list(parts_map.values())
    return plan_data


def _parse_numbered_steps(text: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    pattern = re.compile(
        r"(?:^|\n)\s*(?:Step\s*)?(\d+)[.)]\s*(?:\*\*)?(.+?)(?:\*\*)?(?:\n|$)([\s\S]*?)(?=(?:\n\s*(?:Step\s*)?\d+[.)])|\Z)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        order = int(m.group(1))
        title = m.group(2).strip().strip("*").strip()[:100]
        body = m.group(3).strip()
        substeps = re.findall(r"(?:^|\n)\s*([a-z][.)]\s*.+)", body, re.I | re.M)
        main = re.sub(r"(?:^|\n)\s*[a-z][.)]\s*.+", "", body, flags=re.I | re.M).strip()
        instructions = main if len(main) > 30 else body
        if len(instructions) > 15:
            steps.append({
                "order": order,
                "title": title,
                "instructions": instructions[:1200],
                "substeps": [s.strip() for s in substeps[:6]],
                "tools": [],
                "safety_note": None,
            })
    return steps


def _merge_steps_from_agent(plan_data: dict[str, Any], assistant_content: str) -> dict[str, Any]:
    existing = plan_data.get("steps") or []
    avg_len = sum(len(s.get("instructions", "")) for s in existing) / max(len(existing), 1)
    needs_better = len(existing) < 10 or avg_len < 100

    if not assistant_content:
        return plan_data

    parsed = _parse_numbered_steps(assistant_content)
    if parsed and (needs_better or len(parsed) >= len(existing)):
        plan_data["steps"] = parsed
        return plan_data

    if needs_better:
        paras = [p.strip() for p in re.split(r"\n\s*\n", assistant_content) if len(p.strip()) > 40]
        if len(paras) >= 4:
            plan_data["steps"] = [
                {
                    "order": i + 1,
                    "title": f"Step {i + 1}",
                    "instructions": para,
                    "substeps": [],
                    "tools": [],
                    "safety_note": None,
                }
                for i, para in enumerate(paras[:18])
            ]
    return plan_data


async def _llm_synthesize(context: str) -> dict[str, Any] | None:
    base_url = normalize_ollama_base_url(settings.ollama_base_url)
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="ollama",
        timeout=settings.ollama_timeout_seconds,
    )
    root = base_url.rstrip("/")[:-3] if base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            tags = (await http.get(f"{root}/api/tags")).json()
        available = [m["name"] for m in tags.get("models", []) if m.get("name")]
    except Exception:
        available = []
    model = resolve_ollama_model(settings.ollama_model, available) or settings.ollama_model

    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYNTHESIS_PROMPT},
                    {"role": "user", "content": context},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            return json.loads(raw)
        except (json.JSONDecodeError, Exception):
            if attempt == 1:
                return None
    return None


def _fallback_from_tools(
    tool_trace: list[dict[str, Any]],
    query: str,
    assistant_content: str,
) -> dict[str, Any]:
    title = query[:80] if query else "DIY Project"
    steps: list[dict[str, Any]] = []
    tips: list[str] = []

    for entry in tool_trace:
        tool = entry.get("tool", "")
        try:
            data = json.loads(entry.get("result") or "{}")
        except json.JSONDecodeError:
            continue
        if tool in ("search_diy_guides", "search_detailed_instructions"):
            if data.get("safety_reminder"):
                tips.append(data["safety_reminder"])

    parsed = _parse_numbered_steps(assistant_content)
    if parsed:
        steps = parsed
    elif assistant_content:
        paras = [p.strip() for p in re.split(r"\n\s*\n", assistant_content) if len(p.strip()) > 30]
        steps = [
            {"order": i + 1, "title": f"Step {i + 1}", "instructions": p, "substeps": [], "tools": [], "safety_note": None}
            for i, p in enumerate(paras[:18])
        ]
    # Never use raw web search snippets as steps — they are ads/spam, not instructions

    plan_data = {
        "mode": "plan",
        "title": title,
        "summary": assistant_content[:400] if assistant_content else f"Plan for: {query}",
        "difficulty": "moderate",
        "estimated_time": "Varies",
        "estimated_cost": "See parts list",
        "tools_needed": [],
        "steps": steps,
        "parts": [],
        "tips": tips[:5],
    }
    return _merge_parts_from_trace(plan_data, tool_trace)


def _dict_to_plan(data: dict[str, Any], tool_trace: list[dict[str, Any]] | None = None) -> DIYPlanOut:
    if data.get("mode") == "error" or data.get("error"):
        return DIYPlanOut(
            mode="error",
            error=data.get("error") or "Could not build plan",
            title=data.get("title", ""),
            summary=data.get("summary", ""),
        )

    steps = []
    for s in data.get("steps") or []:
        steps.append(
            StepOut(
                order=int(s.get("order") or len(steps) + 1),
                title=s.get("title", ""),
                instructions=s.get("instructions", ""),
                safety_note=s.get("safety_note"),
                tools=normalize_tools_list(s.get("tools") or []),
                substeps=list(s.get("substeps") or []),
            )
        )

    search_urls: dict[tuple[str, str], str] = {}
    if tool_trace:
        for entry in tool_trace:
            if entry.get("tool") != "search_part_at_store":
                continue
            try:
                td = json.loads(entry.get("result") or "{}")
            except json.JSONDecodeError:
                continue
            pn = td.get("part_name", "")
            for opt in td.get("options") or []:
                search_urls[(pn, opt.get("store", ""))] = td.get("search_url", "")

    parts = []
    for p in data.get("parts") or []:
        part_name = p.get("name", "")
        store_opts = []
        for o in p.get("store_options") or []:
            store = o.get("store", "")
            search_url = search_urls.get((part_name, store), "")
            fixed = _store_option_from_tool({**o, "product_url": o.get("product_url", "")}, part_name, search_url)
            store_opts.append(StoreOptionOut(**fixed))
        parts.append(
            PartOut(
                name=part_name,
                quantity=p.get("quantity", "1"),
                notes=p.get("notes", ""),
                category=p.get("category", "material"),
                store_options=store_opts,
            )
        )

    difficulty = data.get("difficulty", "moderate")
    if difficulty not in DIFFICULTY_LABELS:
        difficulty = "moderate"

    return DIYPlanOut(
        mode=data.get("mode", "plan"),
        title=data.get("title", ""),
        summary=data.get("summary", ""),
        difficulty=difficulty,
        estimated_time=data.get("estimated_time", ""),
        estimated_cost=data.get("estimated_cost", ""),
        tools_needed=normalize_tools_list(data.get("tools_needed") or []),
        steps=steps,
        parts=parts,
        tips=list(data.get("tips") or []),
        error=data.get("error"),
    )


def _default_summary(plan: DIYPlanOut) -> str:
    if plan.error:
        return plan.error
    if plan.summary:
        return plan.summary
    if plan.title:
        return f"DIY plan ready: {plan.title}."
    return "Your DIY plan is ready."
