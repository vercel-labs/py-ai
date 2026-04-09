"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import agent as agent_
import fastapi
import fastapi.middleware.cors
import fastapi.responses
import pydantic
import storage

import ai

app = fastapi.FastAPI(
    title="py-ai-fastapi-chat",
    description="Chat demo using Python Vercel AI SDK",
)

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


file_storage = storage.FileStorage()


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[ai.ai_sdk_ui.UIMessage]
    session_id: str | None = None


@app.post("/chat")
async def chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    messages = ai.ai_sdk_ui.to_messages(request.messages)
    session_id = request.session_id or "default"
    checkpoint_key = f"checkpoint:{session_id}"

    checkpoint = None
    saved = await file_storage.get(checkpoint_key)
    if saved:
        checkpoint = ai.Checkpoint.model_validate(saved)

    durability = ai.EventLogProvider(checkpoint)
    result = agent_.chat_agent.run(agent_.MODEL, messages, durability=durability)

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in ai.ai_sdk_ui.to_sse_stream(result):
            yield chunk

        # Persist checkpoint so interrupted runs (approval hooks with
        # interrupt_loop=True) can resume on re-entry.  Clean up when
        # the run completes without pending hooks.
        cp = durability.checkpoint()
        if cp.steps and not cp.hooks:
            # Steps recorded but no hooks resolved — the run was likely
            # interrupted by an approval hook.  Save for replay.
            await file_storage.put(checkpoint_key, cp.model_dump())
        else:
            await file_storage.delete(checkpoint_key)

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=ai.ai_sdk_ui.UI_MESSAGE_STREAM_HEADERS,
    )
