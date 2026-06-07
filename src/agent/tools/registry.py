"""Tool definitions and execution for the DIY planning agent."""

from __future__ import annotations

import json
from typing import Any

from src.agent.tools import guides, parts, stores
from src.providers import web_search

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_diy_guides",
            "description": "Search for DIY tutorials, repair guides, and safety warnings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string"},
                    "max_results": {"type": "integer", "default": 8},
                },
                "required": ["project"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_detailed_instructions",
            "description": (
                "Deep search for comprehensive step-by-step instructions with measurements and tool lists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string"},
                    "max_results": {"type": "integer", "default": 8},
                },
                "required": ["project"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_parts",
            "description": "Identify exact parts with specs, model numbers, and SKUs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_name": {"type": "string"},
                    "project": {"type": "string", "default": ""},
                    "specs": {"type": "string", "description": "Size, type, material, model"},
                    "max_results": {"type": "integer", "default": 8},
                },
                "required": ["part_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_part_at_store",
            "description": (
                "Find the EXACT product at a store. Returns exact product name, product page URL, "
                "price hint. Always pass specs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_name": {"type": "string"},
                    "store": {"type": "string"},
                    "specs": {"type": "string"},
                    "max_results": {"type": "integer", "default": 6},
                },
                "required": ["part_name", "store"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_stores",
            "description": "List configured hardware stores.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Generic search for techniques, codes, product IDs, or video guides.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 6},
                },
                "required": ["query"],
            },
        },
    },
]


async def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool by name and return JSON string result."""
    try:
        if name == "search_diy_guides":
            result = await guides.search_diy_guides(**arguments)
        elif name == "search_detailed_instructions":
            result = await guides.search_detailed_instructions(**arguments)
        elif name == "search_parts":
            result = await parts.search_parts(**arguments)
        elif name == "search_part_at_store":
            result = await parts.search_part_at_store(**arguments)
        elif name == "list_stores":
            result = stores.list_available_stores()
        elif name == "search_web":
            max_results = min(max(arguments.get("max_results", 6), 1), 10)
            results = await web_search.search_web_async(arguments["query"], max_results=max_results)
            result = {"query": arguments["query"], "results": results}
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": str(exc), "tool": name, "arguments": arguments}

    return json.dumps(result, default=str)
