"""Chat route — streams LLM responses via the AI SDK UI protocol."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import vercel_ai_sdk as ai
from vercel_ai_sdk.ai_sdk_ui import (
    UI_MESSAGE_STREAM_HEADERS,
    UIMessage,
    to_messages,
    to_sse_stream,
)

from ..agent import TOOLS, get_llm, graph
from ..storage import FileStorage

router = APIRouter()
storage = FileStorage()


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    messages: list[UIMessage]
    session_id: str | None = None


@router.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests and stream responses."""
    messages = to_messages(request.messages)
    session_id = request.session_id or "default"
    checkpoint_key = f"checkpoint:{session_id}"

    llm = get_llm()

    # Checkpoints resume an *interrupted* run (e.g. a hook that needed
    # user input in serverless mode).  Each normal chat turn is a fresh
    # run — the frontend carries the full message history — so we only
    # load a checkpoint when one was saved from a previous incomplete run.
    saved = await storage.get(checkpoint_key)
    checkpoint = ai.Checkpoint.deserialize(saved) if saved else None

    result = ai.run(graph, llm, messages, TOOLS, checkpoint=checkpoint)

    async def stream_response():
        async for chunk in to_sse_stream(result):
            yield chunk

        # If the run completed (no pending hooks), clear the checkpoint
        # so the next request starts fresh.  If hooks are pending, save
        # the checkpoint so the next request can resume from here.
        if result.pending_hooks:
            await storage.put(checkpoint_key, result.checkpoint.serialize())
        else:
            await storage.delete(checkpoint_key)

    return StreamingResponse(
        stream_response(),
        headers=UI_MESSAGE_STREAM_HEADERS,
    )
