"""Chat route — streams LLM responses via the AI SDK UI protocol."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import fastapi
import fastapi.responses
import pydantic

import vercel_ai_sdk as ai
import vercel_ai_sdk.ai_sdk_ui

from .. import agent, storage

router = fastapi.APIRouter()
file_storage = storage.FileStorage()


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[ai.ai_sdk_ui.UIMessage]
    session_id: str | None = None


@router.post("/chat")
async def chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    messages = ai.ai_sdk_ui.to_messages(request.messages)
    session_id = request.session_id or "default"
    checkpoint_key = f"checkpoint:{session_id}"

    llm = agent.get_llm()

    # Checkpoints resume an *interrupted* run (e.g. a hook that needed
    # user input in serverless mode).  Each normal chat turn is a fresh
    # run — the frontend carries the full message history — so we only
    # load a checkpoint when one was saved from a previous incomplete run.
    saved = await file_storage.get(checkpoint_key)
    checkpoint = ai.Checkpoint.model_validate(saved) if saved else None

    result = ai.run(agent.graph, llm, messages, agent.TOOLS, checkpoint=checkpoint)

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in ai.ai_sdk_ui.to_sse_stream(result):
            yield chunk

        # If the run completed (no pending hooks), clear the checkpoint
        # so the next request starts fresh.  If hooks are pending, save
        # the checkpoint so the next request can resume from here.
        if result.pending_hooks:
            await file_storage.put(checkpoint_key, result.checkpoint.model_dump())
        else:
            await file_storage.delete(checkpoint_key)

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=ai.ai_sdk_ui.UI_MESSAGE_STREAM_HEADERS,
    )
