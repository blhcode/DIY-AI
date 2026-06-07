"""Inventory extraction helpers — relevance filtering and normalization."""

from __future__ import annotations

import re
from typing import Any

# Obvious junk from bad web-scrape / search noise
_JUNK_RE = re.compile(
    r"\b(soccer|football|basketball|tennis|cricket|xbox|playstation|iphone|"
    r"android|laptop|toy|doll|lego|nintendo|beer|wine|coffee|pizza|"
    r"pet food|dog food|cat litter|nappy|diaper)\b",
    re.I,
)

_STOP = frozenset({
    "the", "and", "for", "with", "from", "buy", "shop", "online", "australia",
    "your", "this", "that", "how", "make", "build", "diy", "step", "guide",
})


_TOOL_RE = re.compile(
    r"\b(?:"
    r"circular saw|miter saw|mitre saw|jigsaw|reciprocating saw|table saw|"
    r"power drill|cordless drill|impact driver|angle grinder|sander|router|"
    r"nail gun|caulk gun|screw gun|heat gun|spray gun|staple gun|"
    r"hammer|mallet|drill|screwdriver|spanner|wrench|pliers|multigrip|"
    r"socket|ratchet|shifter|adjustable spanner|shifting spanner|"
    r"spirit level|bubble level|laser level|try square|combination square|"
    r"framing square|speed square|marking gauge|tape measure|measuring tape|"
    r"chisel|plane|file|rasp|clamp|g-clamp|bar clamp|vice|vise|"
    r"utility knife|stanley knife|tin snips|wire stripper|"
    r"step ladder|stepladder|ladder|workbench|"
    r"safety glasses|goggles|earmuff|ear muff|respirator|dust mask|"
    r"work gloves|gloves|torch|flashlight|multimeter|"
    r"tin snips|hole saw|paddle bit|forstner bit"
    r")\b",
    re.I,
)

# Consumables / parts you buy and install — NOT tools
_MATERIAL_RE = re.compile(
    r"\b(?:"
    r"screws?|nails?|bolts?|nuts?|washers?|anchors?|dowels?|biscuits?|"
    r"hinges?|brackets?|carriage bolts?|coach bolts?|tek screws?|"
    r"timber|pine|hardwood|plywood|mdf|particle board|osb|"
    r"decking|batten|stud|plank|board|lumber|post|rail|"
    r"paint|primer|undercoat|topcoat|stain|varnish|oil|enamel|"
    r"sealant|caulk|silicone|adhesive|glue|epoxy|putty|"
    r"concrete|cement|mortar|grout|sand|gravel|aggregate|"
    r"pipe|fitting|elbow|tee|coupling|flange|cartridge|"
    r"o-ring|gasket|washer kit|tap washer|mixer cartridge|"
    r"flashing|roofing|insulation|batts?|weed mat|landscape fabric|"
    r"mailbox|letterbox|house numbers?|street numbers?|"
    r"exterior screws?|deck screws?|wood screws?|galvanised|galvanized|"
    r"treated pine|h3|h4|resene|wood filler|gap filler|"
    r"sandpaper|abrasive|mesh|tape|masking tape|duct tape|"
    r"brackets?|strap|tie|connector|joist hanger"
    r")\b",
    re.I,
)

# Explicit tool phrases that contain material words
_TOOL_PHRASES = re.compile(
    r"\b(?:"
    r"nail gun|caulk gun|screw gun|heat gun|spray gun|staple gun|"
    r"phillips screwdriver|flathead screwdriver|socket set|"
    r"drill bit|paddle bit|forstner bit|hole saw|"
    r"paint brush|paint roller|roller tray|drop sheet"
    r")\b",
    re.I,
)


def classify_item_kind(name: str, default: str = "material") -> str:
    """Classify as 'tool' (reusable equipment) or 'material' (buy/install/consume)."""
    lower = name.lower().strip()
    if not lower:
        return default
    if _TOOL_PHRASES.search(lower):
        return "tool"
    is_material = bool(_MATERIAL_RE.search(lower))
    is_tool = bool(_TOOL_RE.search(lower))
    if is_material and not is_tool:
        return "material"
    if is_tool and not is_material:
        return "tool"
    if is_material and is_tool:
        # e.g. "drill screws" unlikely; favour material if screws/nails/timber present
        if re.search(r"\b(screws?|nails?|timber|pine|paint|pipe|bracket)\b", lower):
            return "material"
        return "tool"
    return default


def item_to_label(item: Any) -> str:
    """Turn a tool/material entry (string or dict) into a display label."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        name = str(item.get("name") or item.get("item") or item.get("title") or "").strip()
        specs = str(item.get("specs") or item.get("notes") or item.get("description") or "").strip()
        if name and specs:
            return f"{name} ({specs})"
        return name
    if item is None:
        return ""
    return str(item).strip()


def normalize_tools_list(tools: list[Any]) -> list[str]:
    """Coerce LLM tool entries to plain strings; dedupe preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in tools:
        label = item_to_label(raw)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def normalize_item(raw: dict[str, Any], *, kind: str = "material") -> dict[str, str] | None:
    """Normalize LLM inventory entry; return None if no usable name."""
    name = (
        str(raw.get("name") or raw.get("item") or raw.get("title") or "")
    ).strip()
    if not name:
        return None
    specs = str(raw.get("specs") or raw.get("notes") or raw.get("description") or "").strip()
    qty = str(raw.get("quantity") or "1").strip() or "1"
    resolved_kind = classify_item_kind(name, default=kind)
    return {"name": name, "quantity": qty, "specs": specs, "kind": resolved_kind}


def merge_items(*lists: list[dict[str, str]]) -> list[dict[str, str]]:
    """Dedupe by normalized name, prefer entries with specs."""
    out: dict[str, dict[str, str]] = {}
    for items in lists:
        for raw in items:
            if not raw.get("name"):
                continue
            key = re.sub(r"\s+", " ", raw["name"].strip().lower())
            if key not in out:
                out[key] = dict(raw)
            elif raw.get("specs") and not out[key].get("specs"):
                out[key]["specs"] = raw["specs"]
            if raw.get("quantity") and out[key].get("quantity") == "1":
                out[key]["quantity"] = raw["quantity"]
    return list(out.values())


def product_title_matches(part_name: str, title: str) -> bool:
    """True when a store search result plausibly matches the part we searched for."""
    part_words = {
        w for w in re.findall(r"\w{3,}", part_name.lower()) if w not in _STOP
    }
    if not part_words:
        return True
    title_lower = title.lower()
    hits = sum(1 for w in part_words if w in title_lower)
    return hits >= max(1, (len(part_words) + 1) // 2)


def filter_relevant_items(items: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    """Drop obvious junk from bad search matches; keep all plausible DIY items."""
    kept: list[dict[str, str]] = []
    for item in items:
        name = item.get("name", "").strip()
        if not name or len(name) < 3:
            continue
        if _JUNK_RE.search(name):
            continue
        kept.append(item)
    return kept
