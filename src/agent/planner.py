"""DIY planning AI agent with Ollama tool use."""

from __future__ import annotations

import json
from datetime import date
from typing import Any
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools.registry import TOOL_DEFINITIONS, execute_tool
from src.config import settings
from src.ollama_utils import normalize_ollama_base_url, resolve_ollama_model

MAX_TOOL_ROUNDS = 16


class DIYPlannerAgent:
    def __init__(self) -> None:
        base_url = normalize_ollama_base_url(settings.ollama_base_url)
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=settings.ollama_timeout_seconds,
        )
        self.model = settings.ollama_model
        self._resolved_model: str | None = None

    @staticmethod
    def _ollama_root() -> str:
        base = normalize_ollama_base_url(settings.ollama_base_url).rstrip("/")
        if base.endswith("/v1"):
            return base[:-3]
        return base

    async def _fetch_available_models(self) -> list[str]:
        root = self._ollama_root()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{root}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    async def _resolve_model(self) -> str | None:
        if self._resolved_model:
            return self._resolved_model
        available = await self._fetch_available_models()
        resolved = resolve_ollama_model(self.model, available)
        if resolved:
            self._resolved_model = resolved
        return resolved

    async def check_connection(self) -> dict[str, Any]:
        """Verify Ollama is reachable and the configured model is available."""
        root = self._ollama_root()
        host = urlparse(root).netloc or root
        try:
            available = await self._fetch_available_models()
        except Exception as exc:
            return {
                "connected": False,
                "model_available": False,
                "host": host,
                "model": self.model,
                "resolved_model": None,
                "error": f"Ollama unreachable at {settings.ollama_base_url}: {exc}",
                "available_models": [],
            }

        resolved = resolve_ollama_model(self.model, available)
        if resolved:
            self._resolved_model = resolved

        return {
            "connected": True,
            "model_available": resolved is not None,
            "host": host,
            "model": self.model,
            "resolved_model": resolved,
            "error": None
            if resolved
            else (
                f"Model '{self.model}' not found. Available: {', '.join(available) or 'none'}. "
                f"Set OLLAMA_MODEL in diy.env (e.g. llama3.1:8b)"
            ),
            "available_models": available,
        }

    async def _model_for_request(self) -> str:
        resolved = await self._resolve_model()
        if not resolved:
            available = await self._fetch_available_models()
            raise ValueError(
                f"Model '{self.model}' not found on Ollama. "
                f"Available: {', '.join(available) or 'none'}. "
                f"Update OLLAMA_MODEL in diy.env"
            )
        return resolved

    async def chat(
        self,
        messages: list[dict[str, str]],
        user_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run agent loop: LLM may call tools until it produces a final answer.
        Returns assistant message and tool trace for synthesis.
        """
        today = date.today().isoformat()
        system = SYSTEM_PROMPT + f"\n\nToday's date: {today}."
        system += (
            f"\nUser region: {settings.locale_hint} ({settings.default_country}). "
            f"Use {settings.currency} for cost estimates."
        )
        stores = settings.stores_list
        preferred = (user_context or {}).get("preferred_stores") or []
        if preferred:
            stores = [s for s in stores if s in preferred or any(p.lower() in s.lower() for p in preferred)]
            if not stores:
                stores = preferred
            system += f"\nUser wants to shop ONLY at: {', '.join(preferred)}. Search parts at these stores only."
        if stores:
            system += f"\nStores to search for every part: {', '.join(stores)}."
        if settings.default_location:
            system += f"\nUser location for store searches: {settings.default_location}."
        if user_context:
            system += f"\nUser context: {json.dumps(user_context)}"

        api_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        tool_trace: list[dict[str, Any]] = []
        model = await self._model_for_request()

        for _ in range(MAX_TOOL_ROUNDS):
            response = await self.client.chat.completions.create(
                model=model,
                messages=api_messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
            choice = response.choices[0]
            assistant_msg = choice.message

            if assistant_msg.tool_calls:
                api_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_msg.tool_calls
                        ],
                    }
                )
                for tc in assistant_msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    result = await execute_tool(tc.function.name, args)
                    tool_trace.append(
                        {
                            "tool": tc.function.name,
                            "arguments": args,
                            "result": result,
                            "result_preview": result[:500],
                        }
                    )
                    api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
                continue

            return {
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_trace": tool_trace,
                "api_messages": api_messages,
            }

        return {
            "role": "assistant",
            "content": (
                "I've gathered a lot of information but hit the research limit. "
                "Try asking about one part of the project at a time."
            ),
            "tool_trace": tool_trace,
            "api_messages": api_messages,
        }
