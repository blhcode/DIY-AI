"""JSON-serializable DIY plan structures for the API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StoreOptionOut:
    store: str
    price_hint: str
    url: str
    snippet: str
    confidence: str = "medium"
    link_type: str = "search"
    product_url: str = ""
    exact_product_name: str = ""
    product_id: str = ""


@dataclass
class PartOut:
    name: str
    quantity: str
    notes: str
    category: str = "material"  # material | tool | consumable
    store_options: list[StoreOptionOut] = field(default_factory=list)


@dataclass
class StepOut:
    order: int
    title: str
    instructions: str
    safety_note: str | None = None
    tools: list[str] = field(default_factory=list)
    substeps: list[str] = field(default_factory=list)


@dataclass
class DIYPlanOut:
    mode: str = "plan"
    title: str = ""
    summary: str = ""
    difficulty: str = "moderate"
    estimated_time: str = ""
    estimated_cost: str = ""
    tools_needed: list[str] = field(default_factory=list)
    steps: list[StepOut] = field(default_factory=list)
    parts: list[PartOut] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
