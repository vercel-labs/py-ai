"""Hooks: live resolution, cancellation, pre-registration, schema validation."""

import asyncio
from typing import Any, ClassVar

import pydantic
import pytest

import vercel_ai_sdk as ai

from ..conftest import MOCK_MODEL, mock_llm, text_msg


@ai.hook
class Confirmation(pydantic.BaseModel):
    approved: bool
    reason: str = ""


@ai.hook
class CancellingConfirmation(pydantic.BaseModel):
    cancels_future: ClassVar[bool] = True
    approved: bool
    reason: str = ""


# -- Hook.resolve() with live future (long-running mode) -------------------


@pytest.mark.asyncio
async def test_resolve_live_future() -> None:
    """In long-running mode, Hook.resolve() unblocks the awaiting coroutine."""
    resolved_value = None
    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> None:
        nonlocal resolved_value
        await ai.stream_step(agent.model, msgs)
        result = await Confirmation.create("confirm_1")  # type: ignore[attr-defined]
        resolved_value = result

    mock_llm([[text_msg("OK")]])
    # Confirmation.cancels_future=False -> long-running mode
    run_result = my_agent.run(ai.make_messages(user="go"))

    collected = []
    async for msg in run_result:
        collected.append(msg)
        # When we see the pending hook message, resolve it
        if any(isinstance(p, ai.HookPart) and p.status == "pending" for p in msg.parts):
            Confirmation.resolve(  # type: ignore[attr-defined]
                "confirm_1", {"approved": True, "reason": "looks good"}
            )

    assert resolved_value is not None
    assert resolved_value.approved is True
    assert resolved_value.reason == "looks good"


# -- Hook.cancel() --------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_live_hook() -> None:
    """Hook.cancel() cancels the future, causing CancelledError in graph."""
    was_cancelled = False
    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> None:
        nonlocal was_cancelled
        await ai.stream_step(agent.model, msgs)
        try:
            await Confirmation.create("cancel_me")  # type: ignore[attr-defined]
        except asyncio.CancelledError:
            was_cancelled = True

    mock_llm([[text_msg("OK")]])
    run_result = my_agent.run(ai.make_messages(user="go"))

    async for msg in run_result:
        if any(isinstance(p, ai.HookPart) and p.status == "pending" for p in msg.parts):
            await Confirmation.cancel("cancel_me", reason="denied")  # type: ignore[attr-defined]

    assert was_cancelled


# -- Hook.cancel() on non-existent label raises ----------------------------


@pytest.mark.asyncio
async def test_cancel_nonexistent_raises() -> None:
    with pytest.raises(ValueError, match="No pending hook"):
        await Confirmation.cancel("does_not_exist_xyz")  # type: ignore[attr-defined]


# -- Pre-registration (serverless re-entry) --------------------------------


@pytest.mark.asyncio
async def test_pre_registered_resolution_consumed() -> None:
    """Pre-registered resolution is consumed by Hook.create() without suspending."""
    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> Any:
        await ai.stream_step(agent.model, msgs)
        result = await Confirmation.create("pre_reg_1")  # type: ignore[attr-defined]
        return result

    # Pre-register BEFORE run
    Confirmation.resolve("pre_reg_1", {"approved": True})  # type: ignore[attr-defined]

    mock_llm([[text_msg("OK")]])
    run_result = my_agent.run(ai.make_messages(user="go"))
    [m async for m in run_result]

    # Should have completed with no pending hooks
    assert len(run_result.pending_hooks) == 0
    # Hook event should be in checkpoint
    assert any(h.label == "pre_reg_1" for h in run_result.checkpoint.hooks)


# -- Schema validation on resolve -----------------------------------------


def test_resolve_validates_schema() -> None:
    """resolve() with invalid data raises from pydantic validation."""
    # 'approved' is required bool, passing string should raise
    with pytest.raises(pydantic.ValidationError):
        Confirmation.resolve("schema_test", {"approved": "not_a_bool"})  # type: ignore[attr-defined]


# -- Resolved hook emits message -------------------------------------------


@pytest.mark.asyncio
async def test_resolved_hook_emits_message() -> None:
    """After resolution, a 'resolved' HookPart message is emitted."""
    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> None:
        await ai.stream_step(agent.model, msgs)
        await Confirmation.create("emit_test")  # type: ignore[attr-defined]

    mock_llm([[text_msg("OK")]])
    run_result = my_agent.run(ai.make_messages(user="go"))

    msgs = []
    async for msg in run_result:
        msgs.append(msg)
        if any(isinstance(p, ai.HookPart) and p.status == "pending" for p in msg.parts):
            Confirmation.resolve("emit_test", {"approved": False})  # type: ignore[attr-defined]

    hook_msgs = [
        m
        for m in msgs
        if any(isinstance(p, ai.HookPart) and p.status == "resolved" for p in m.parts)
    ]
    assert len(hook_msgs) == 1
    assert hook_msgs[0].parts[0].resolution == {"approved": False, "reason": ""}  # type: ignore[union-attr]


# -- Hook metadata surfaces in pending message -----------------------------


@pytest.mark.asyncio
async def test_hook_metadata_in_pending() -> None:
    my_agent = ai.agent(model=MOCK_MODEL)

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> None:
        await ai.stream_step(agent.model, msgs)
        await CancellingConfirmation.create(  # type: ignore[attr-defined]
            "meta_test", metadata={"tool": "rm -rf", "path": "/"}
        )

    mock_llm([[text_msg("OK")]])
    run_result = my_agent.run(ai.make_messages(user="go"))
    [m async for m in run_result]

    info = run_result.pending_hooks["meta_test"]
    assert info.metadata == {"tool": "rm -rf", "path": "/"}
