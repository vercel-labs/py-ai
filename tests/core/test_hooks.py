"""Hooks: live resolution, cancellation, pre-registration, schema validation."""

import asyncio
from typing import Any

import pydantic
import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.core.hooks import _live_hooks, _pending_resolutions

from ..conftest import MockLLM, text_msg


@ai.hook
class Confirmation(pydantic.BaseModel):
    approved: bool
    reason: str = ""


# -- Hook.resolve() with live future (long-running mode) -------------------


@pytest.mark.asyncio
async def test_resolve_live_future() -> None:
    """In long-running mode, Hook.resolve() unblocks the awaiting coroutine."""
    resolved_value = None

    async def graph(llm: ai.LanguageModel) -> None:
        nonlocal resolved_value
        await ai.stream_step(llm, ai.make_messages(user="go"))
        result = await Confirmation.create("confirm_1")
        resolved_value = result

    llm = MockLLM([[text_msg("OK")]])
    # Default cancel_on_hooks=False -> long-running mode
    run_result = ai.run(graph, llm)

    collected = []
    async for msg in run_result:
        collected.append(msg)
        # When we see the pending hook message, resolve it
        if any(isinstance(p, ai.HookPart) and p.status == "pending" for p in msg.parts):
            Confirmation.resolve(
                "confirm_1", {"approved": True, "reason": "looks good"}
            )

    assert resolved_value is not None
    assert resolved_value.approved is True
    assert resolved_value.reason == "looks good"
    # The graph completed successfully (resolved_value proves it).
    # Note: pending_hooks is not cleaned up after live resolution --
    # that's a known runtime limitation. The important thing is the
    # graph continued past the hook.


# -- Hook.cancel() --------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_live_hook() -> None:
    """Hook.cancel() cancels the future, causing CancelledError in graph."""
    was_cancelled = False

    async def graph(llm: ai.LanguageModel) -> None:
        nonlocal was_cancelled
        await ai.stream_step(llm, ai.make_messages(user="go"))
        try:
            await Confirmation.create("cancel_me")
        except asyncio.CancelledError:
            was_cancelled = True

    llm = MockLLM([[text_msg("OK")]])
    run_result = ai.run(graph, llm)

    async for msg in run_result:
        if any(isinstance(p, ai.HookPart) and p.status == "pending" for p in msg.parts):
            await Confirmation.cancel("cancel_me", reason="denied")

    assert was_cancelled


# -- Hook.cancel() on non-existent label raises ----------------------------


@pytest.mark.asyncio
async def test_cancel_nonexistent_raises() -> None:
    with pytest.raises(ValueError, match="No pending hook"):
        await Confirmation.cancel("does_not_exist_xyz")


# -- Pre-registration (serverless re-entry) --------------------------------


@pytest.mark.asyncio
async def test_pre_registered_resolution_consumed() -> None:
    """Pre-registered resolution is consumed by Hook.create() without suspending."""

    async def graph(llm: ai.LanguageModel) -> Any:
        await ai.stream_step(llm, ai.make_messages(user="go"))
        result = await Confirmation.create("pre_reg_1")
        return result

    # Pre-register BEFORE run
    Confirmation.resolve("pre_reg_1", {"approved": True})

    llm = MockLLM([[text_msg("OK")]])
    run_result = ai.run(graph, llm)
    msgs = [m async for m in run_result]

    # Should have completed with no pending hooks
    assert len(run_result.pending_hooks) == 0
    # Hook event should be in checkpoint
    assert any(h.label == "pre_reg_1" for h in run_result.checkpoint.hooks)


# -- Schema validation on resolve -----------------------------------------


def test_resolve_validates_schema() -> None:
    """resolve() with invalid data raises from pydantic validation."""
    # 'approved' is required bool, passing string should raise
    with pytest.raises(pydantic.ValidationError):
        Confirmation.resolve("schema_test", {"approved": "not_a_bool"})


# -- Resolved hook emits message -------------------------------------------


@pytest.mark.asyncio
async def test_resolved_hook_emits_message() -> None:
    """After resolution, a 'resolved' HookPart message is emitted."""

    async def graph(llm: ai.LanguageModel) -> None:
        await ai.stream_step(llm, ai.make_messages(user="go"))
        await Confirmation.create("emit_test")

    llm = MockLLM([[text_msg("OK")]])
    run_result = ai.run(graph, llm)

    msgs = []
    async for msg in run_result:
        msgs.append(msg)
        if any(isinstance(p, ai.HookPart) and p.status == "pending" for p in msg.parts):
            Confirmation.resolve("emit_test", {"approved": False})

    hook_msgs = [
        m
        for m in msgs
        if any(isinstance(p, ai.HookPart) and p.status == "resolved" for p in m.parts)
    ]
    assert len(hook_msgs) == 1
    assert hook_msgs[0].parts[0].resolution == {"approved": False, "reason": ""}


# -- Hook metadata surfaces in pending message -----------------------------


@pytest.mark.asyncio
async def test_hook_metadata_in_pending() -> None:
    async def graph(llm: ai.LanguageModel) -> None:
        await ai.stream_step(llm, ai.make_messages(user="go"))
        await Confirmation.create("meta_test", metadata={"tool": "rm -rf", "path": "/"})

    run_result = ai.run(graph, MockLLM([[text_msg("OK")]]), cancel_on_hooks=True)
    msgs = [m async for m in run_result]

    info = run_result.pending_hooks["meta_test"]
    assert info.metadata == {"tool": "rm -rf", "path": "/"}
