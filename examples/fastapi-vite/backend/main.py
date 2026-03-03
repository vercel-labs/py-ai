"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import agent
import fastapi
import fastapi.middleware.cors
import fastapi.responses
import pydantic
import storage

import vercel_ai_sdk as ai
import vercel_ai_sdk.ai_sdk_ui

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


def _has_matching_approval(
    ui_messages: list[ai.ai_sdk_ui.UIMessage],
    pending_hooks: list[str],
) -> bool:
    """True when the incoming messages resolve at least one pending hook.

    Hook labels follow the ``approve_{tool_call_id}`` convention set by
    ``_execute_with_approval`` in the agent graph.
    """
    pending = set(pending_hooks)
    for msg in ui_messages:
        for part in msg.parts:
            state = getattr(part, "state", None)
            tcid = getattr(part, "tool_call_id", None)
            if (
                state == "approval-responded"
                and tcid is not None
                and f"approve_{tcid}" in pending
            ):
                return True
    return False


@app.post("/chat")
async def chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    messages = ai.ai_sdk_ui.to_messages(request.messages)
    session_id = request.session_id or "default"
    checkpoint_key = f"checkpoint:{session_id}"

    llm = agent.get_llm()

    # Only load a checkpoint when this request is actually resuming
    # an interrupted run — i.e. the frontend is sending back an
    # approval response that matches a pending hook.  Otherwise
    # discard stale checkpoints so fresh turns aren't poisoned.
    checkpoint = None
    saved = await file_storage.get(checkpoint_key)
    if saved:
        pending = saved.get("pending_hooks", [])
        if _has_matching_approval(request.messages, pending):
            checkpoint = ai.Checkpoint.model_validate(saved["checkpoint"])
            # The frontend sends the full message history including the
            # assistant message from the interrupted run.  The checkpoint
            # will replay that same step, so strip the trailing assistant
            # message to avoid sending a duplicate tool_use to the LLM.
            if messages and messages[-1].role == "assistant":
                messages = messages[:-1]
        else:
            await file_storage.delete(checkpoint_key)

    result = ai.run(
        agent.graph,
        llm,
        messages,
        agent.TOOLS,
        checkpoint=checkpoint,
        cancel_on_hooks=True,
    )

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in ai.ai_sdk_ui.to_sse_stream(result):
            yield chunk

        # Save checkpoint + pending hook labels so the next request
        # can decide whether it's a resume or a fresh turn.
        if result.pending_hooks:
            await file_storage.put(
                checkpoint_key,
                {
                    "checkpoint": result.checkpoint.model_dump(),
                    "pending_hooks": list(result.pending_hooks.keys()),
                },
            )
        else:
            await file_storage.delete(checkpoint_key)

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=ai.ai_sdk_ui.UI_MESSAGE_STREAM_HEADERS,
    )
