"""FastAPI server for DIY AI."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent.planner import DIYPlannerAgent
from src.agent.synthesizer import synthesize_plan
from src.config import settings

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
APP_VERSION = "1.0.0"

app = FastAPI(title="DIY AI", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = DIYPlannerAgent()
logger = logging.getLogger(__name__)

PLAN_TIMEOUT_SECONDS = 360


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    preferred_stores: list[str] | None = None


class ChatResponse(BaseModel):
    message: ChatMessage
    plan: dict[str, Any] | None = None


def _ollama_setup_message(status: dict[str, Any]) -> str:
    if not status["connected"]:
        return (
            "Ollama is unreachable. Edit diy.env and set OLLAMA_BASE_URL "
            f"(currently {status.get('host', 'unknown')}). "
            "Ensure Ollama is running: ollama serve"
        )
    if not status["model_available"]:
        models = ", ".join(status.get("available_models") or []) or "none"
        return (
            f"Model '{status['model']}' not found on Ollama. "
            f"Set OLLAMA_MODEL in diy.env to one of: {models}. "
            f"(Run `ollama list` on your Ollama machine.)"
        )
    return ""


@app.get("/api/health")
async def health():
    status = await _agent.check_connection()
    return {
        "status": "ok" if status["connected"] and status["model_available"] else "degraded",
        "version": APP_VERSION,
        "mode": "ollama_agent",
        "ollama_url": status["host"],
        "ollama_model": status["model"],
        "ollama_connected": status["connected"],
        "ollama_model_available": status["model_available"],
        "requires_api_keys": False,
        "stores": settings.stores_list,
        "country": settings.default_country,
        "locale": settings.locale_hint,
        "currency": settings.currency,
        "location": settings.default_location or None,
        "error": status.get("error"),
    }


@app.get("/api/stores")
async def stores():
    return {
        "stores": settings.stores_list,
        "country": settings.default_country,
        "locale": settings.locale_hint,
        "currency": settings.currency,
        "location": settings.default_location or None,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages required")
    if req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    status = await _agent.check_connection()
    setup_msg = _ollama_setup_message(status)
    if setup_msg:
        raise HTTPException(status_code=503, detail=setup_msg)

    query = req.messages[-1].content

    agent_result: dict[str, Any] = {"role": "assistant", "content": "", "tool_trace": []}

    try:
        plan_obj, summary = await asyncio.wait_for(
            synthesize_plan(
                agent_result,
                query,
                preferred_stores=req.preferred_stores,
            ),
            timeout=PLAN_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Request timed out after {PLAN_TIMEOUT_SECONDS}s. "
                "Pick one store in the dropdown to speed things up, or try a shorter question."
            ),
        ) from None
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach Ollama at {settings.ollama_base_url}. Is it running? ({exc})",
        ) from exc
    except Exception as exc:
        logger.exception("plan build failed")
        raise HTTPException(
            status_code=504,
            detail=f"Plan build failed: {exc}. Pick one store and try again.",
        ) from exc

    plan_dict = plan_obj.to_dict()

    return ChatResponse(
        message=ChatMessage(role="assistant", content=summary),
        plan=plan_dict,
    )


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
