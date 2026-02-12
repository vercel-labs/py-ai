"""
Tests for checkpoint-based replay and hook cancellation.
"""

import asyncio
from collections.abc import AsyncGenerator

import pydantic
import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.core import messages


# ── Mock LLM ──────────────────────────────────────────────────────


class MockLLM(ai.LanguageModel):
    """A mock LLM that counts calls and returns pre-defined responses."""

    def __init__(self, responses: list[list[messages.Message]]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.call_count = 0

    async def stream(
        self,
        messages: list[messages.Message],
        tools: list[ai.Tool] | None = None,
    ) -> AsyncGenerator[messages.Message, None]:
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockLLM: no more responses configured")

        self.call_count += 1
        response_sequence = self._responses[self._call_index]
        self._call_index += 1

        for msg in response_sequence:
            yield msg


# ── Test tools and hooks ──────────────────────────────────────────


@ai.tool
async def add_one(x: int) -> int:
    """Add one to x."""
    return x + 1


@ai.hook
class Approval(pydantic.BaseModel):
    granted: bool


# ── Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_step_replay_skips_llm():
    """Steps replayed from checkpoint don't call the LLM."""

    async def graph(llm: ai.LanguageModel):
        return await ai.stream_step(
            llm,
            messages=ai.make_messages(system="test", user="hello"),
        )

    # First run: LLM is called
    llm1 = MockLLM(
        [
            [
                messages.Message(
                    id="msg-1",
                    role="assistant",
                    parts=[messages.TextPart(text="Hi there!", state="done")],
                )
            ]
        ]
    )

    result1 = ai.run(graph, llm1)
    msgs1 = [msg async for msg in result1]
    assert llm1.call_count == 1
    assert any("Hi there!" in m.text for m in msgs1)

    # Capture checkpoint
    cp = result1.checkpoint
    assert len(cp.steps) == 1

    # Second run with checkpoint: LLM should NOT be called
    llm2 = MockLLM([])  # no responses — would fail if called
    result2 = ai.run(graph, llm2, checkpoint=cp)
    msgs2 = [msg async for msg in result2]
    assert llm2.call_count == 0
    # Replayed messages are not yielded to the stream (they were already
    # seen by the client on the previous request)
    # The graph still completes successfully though


@pytest.mark.asyncio
async def test_tool_replay_skips_execution():
    """Tool results replayed from checkpoint skip actual execution."""

    execution_count = 0

    @ai.tool
    async def counting_tool(x: int) -> int:
        """Counts how many times it's called."""
        nonlocal execution_count
        execution_count += 1
        return x + 1

    async def graph(llm: ai.LanguageModel):
        result = await ai.stream_step(llm, ai.make_messages(system="t", user="go"))
        if result.tool_calls:
            await asyncio.gather(
                *(
                    ai.execute_tool(tc, message=result.last_message)
                    for tc in result.tool_calls
                )
            )
        return result

    # First run
    llm1 = MockLLM(
        [
            [
                messages.Message(
                    id="msg-1",
                    role="assistant",
                    parts=[
                        messages.ToolPart(
                            tool_call_id="tc-1",
                            tool_name="counting_tool",
                            tool_args='{"x": 5}',
                            status="pending",
                            state="done",
                        )
                    ],
                )
            ]
        ]
    )

    result1 = ai.run(graph, llm1)
    [msg async for msg in result1]
    assert execution_count == 1

    cp = result1.checkpoint
    assert len(cp.tools) == 1
    assert cp.tools[0].tool_call_id == "tc-1"
    assert cp.tools[0].result == 6

    # Second run with checkpoint: tool should NOT execute again
    execution_count = 0
    result2 = ai.run(graph, MockLLM([]), checkpoint=cp)
    [msg async for msg in result2]
    assert execution_count == 0


@pytest.mark.asyncio
async def test_hook_cancellation_with_pending_hooks():
    """Unresolved hooks are cancelled; pending_hooks is populated on RunResult."""

    async def graph(llm: ai.LanguageModel):
        result = await ai.stream_step(llm, ai.make_messages(system="t", user="go"))
        # Hit a hook — no resolution available, should be cancelled
        approval = await Approval.create("my_approval", metadata={"tool": "test"})
        return approval

    llm = MockLLM(
        [
            [
                messages.Message(
                    id="msg-1",
                    role="assistant",
                    parts=[messages.TextPart(text="OK", state="done")],
                )
            ]
        ]
    )

    result = ai.run(graph, llm, cancel_on_hooks=True)
    msgs = [msg async for msg in result]

    # Hook should be pending
    assert "my_approval" in result.pending_hooks
    info = result.pending_hooks["my_approval"]
    assert info.hook_type == "Approval"
    assert info.metadata == {"tool": "test"}

    # A pending hook message should have been yielded
    hook_msgs = [m for m in msgs if any(isinstance(p, ai.HookPart) for p in m.parts)]
    assert len(hook_msgs) == 1
    assert hook_msgs[0].parts[0].status == "pending"


@pytest.mark.asyncio
async def test_hook_resolution_on_reentry():
    """Hook resolves immediately when resolution is provided on re-entry."""

    async def graph(llm: ai.LanguageModel):
        result = await ai.stream_step(llm, ai.make_messages(system="t", user="go"))
        approval = await Approval.create("my_approval")
        # If we get here, the hook was resolved
        return approval

    response = [
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[messages.TextPart(text="OK", state="done")],
        )
    ]

    # First run: hook is cancelled (no resolution, serverless mode)
    result1 = ai.run(graph, MockLLM([response]), cancel_on_hooks=True)
    [msg async for msg in result1]
    assert "my_approval" in result1.pending_hooks
    cp = result1.checkpoint

    # Second run: pre-register resolution, replay step from checkpoint
    Approval.resolve("my_approval", {"granted": True})
    result2 = ai.run(
        graph,
        MockLLM([]),  # no LLM responses needed — step replays from checkpoint
        checkpoint=cp,
    )
    msgs2 = [msg async for msg in result2]

    # No pending hooks — graph completed
    assert len(result2.pending_hooks) == 0
    # Checkpoint should now include the hook event
    cp2 = result2.checkpoint
    assert len(cp2.hooks) == 1
    assert cp2.hooks[0].label == "my_approval"


@pytest.mark.asyncio
async def test_parallel_hooks_all_collected():
    """Multiple hooks in parallel branches are all collected as pending."""

    async def graph(llm: ai.LanguageModel):
        result = await ai.stream_step(llm, ai.make_messages(system="t", user="go"))

        async def branch_a():
            return await Approval.create("hook_a")

        async def branch_b():
            return await Approval.create("hook_b")

        async with asyncio.TaskGroup() as tg:
            ta = tg.create_task(branch_a())
            tb = tg.create_task(branch_b())

    llm = MockLLM(
        [
            [
                messages.Message(
                    id="msg-1",
                    role="assistant",
                    parts=[messages.TextPart(text="OK", state="done")],
                )
            ]
        ]
    )

    result = ai.run(graph, llm, cancel_on_hooks=True)
    [msg async for msg in result]

    # Both hooks should be pending
    assert "hook_a" in result.pending_hooks
    assert "hook_b" in result.pending_hooks


@pytest.mark.asyncio
async def test_parallel_hooks_resolve_on_reentry():
    """Parallel hooks all resolve when resolutions are provided on re-entry."""

    async def graph(llm: ai.LanguageModel):
        result = await ai.stream_step(llm, ai.make_messages(system="t", user="go"))

        async def branch_a():
            a = await Approval.create("hook_a")
            return a

        async def branch_b():
            b = await Approval.create("hook_b")
            return b

        async with asyncio.TaskGroup() as tg:
            ta = tg.create_task(branch_a())
            tb = tg.create_task(branch_b())

        return ta.result(), tb.result()

    response = [
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[messages.TextPart(text="OK", state="done")],
        )
    ]

    # First run: both hooks cancelled (serverless mode)
    result1 = ai.run(graph, MockLLM([response]), cancel_on_hooks=True)
    [msg async for msg in result1]
    assert len(result1.pending_hooks) == 2
    cp = result1.checkpoint

    # Second run: pre-register resolutions, then replay
    Approval.resolve("hook_a", {"granted": True})
    Approval.resolve("hook_b", {"granted": False})
    result2 = ai.run(
        graph,
        MockLLM([]),
        checkpoint=cp,
    )
    [msg async for msg in result2]
    assert len(result2.pending_hooks) == 0


@pytest.mark.asyncio
async def test_checkpoint_serialization_roundtrip():
    """Checkpoint serializes and deserializes correctly."""
    from vercel_ai_sdk.core.checkpoint import (
        Checkpoint,
        HookEvent,
        StepEvent,
        ToolEvent,
    )

    cp = Checkpoint(
        steps=[
            StepEvent(
                index=0,
                messages=[
                    {
                        "id": "msg-1",
                        "role": "assistant",
                        "parts": [{"type": "text", "text": "hello"}],
                    }
                ],
            )
        ],
        tools=[ToolEvent(tool_call_id="tc-1", result=42)],
        hooks=[HookEvent(label="my_hook", resolution={"granted": True})],
    )

    data = cp.serialize()
    cp2 = Checkpoint.deserialize(data)

    assert len(cp2.steps) == 1
    assert cp2.steps[0].index == 0
    assert len(cp2.tools) == 1
    assert cp2.tools[0].result == 42
    assert len(cp2.hooks) == 1
    assert cp2.hooks[0].label == "my_hook"
