"""Agent default loop, execute_tool, multi-turn, Runtime injection."""

import asyncio

import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.agents.runtime import Runtime
from vercel_ai_sdk.types import messages

from ..conftest import MOCK_MODEL, mock_llm, text_msg, tool_call_msg

# -- Tool definitions for tests --------------------------------------------


@ai.tool
async def double(x: int) -> int:
    """Double a number."""
    return x * 2


@ai.tool
async def concat(a: str, b: str) -> str:
    """Concatenate strings."""
    return a + b


# -- Agent default loop: single turn (no tools) ----------------------------


@pytest.mark.asyncio
async def test_agent_text_only() -> None:
    """Agent default loop with no tool calls returns after one LLM call."""
    my_agent = ai.agent(model=MOCK_MODEL, tools=[double])

    llm = mock_llm([[text_msg("Hello!")]])
    result = my_agent.run(ai.make_messages(user="Hi"))
    msgs = [m async for m in result]
    assert llm.call_count == 1
    assert any(m.text == "Hello!" for m in msgs)


# -- Agent default loop: tool call + follow-up -----------------------------


@pytest.mark.asyncio
async def test_agent_tool_then_text() -> None:
    """Agent default loop calls tool, feeds result back, gets final text."""
    my_agent = ai.agent(model=MOCK_MODEL, tools=[double])

    call1 = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 5}')]
    call2 = [text_msg("The answer is 10.")]
    llm = mock_llm([call1, call2])

    result = my_agent.run(ai.make_messages(user="Double 5"))
    msgs = [m async for m in result]
    assert llm.call_count == 2
    # Tool should have been executed: 5 * 2 = 10
    tool_results = [m for m in msgs if m.role == "tool" and m.tool_results]
    assert len(tool_results) >= 1
    assert tool_results[0].tool_results[0].result == 10


# -- Agent default loop: multiple tool calls in one message ----------------


@pytest.mark.asyncio
async def test_agent_parallel_tools() -> None:
    """LLM returns two tool calls in one message; both execute."""
    my_agent = ai.agent(model=MOCK_MODEL, tools=[double])

    two_tools = messages.Message(
        id="msg-1",
        role="assistant",
        parts=[
            messages.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="double",
                tool_args='{"x": 3}',
                state="done",
            ),
            messages.ToolCallPart(
                tool_call_id="tc-2",
                tool_name="double",
                tool_args='{"x": 7}',
                state="done",
            ),
        ],
    )
    call2 = [text_msg("6 and 14", id="msg-2")]
    llm = mock_llm([[two_tools], call2])

    result = my_agent.run(ai.make_messages(user="Double 3 and 7"))
    msgs = [m async for m in result]
    assert llm.call_count == 2
    # Both tools should have results
    tool_result_msgs = [m for m in msgs if m.role == "tool" and m.tool_results]
    assert len(tool_result_msgs) >= 1


# -- Agent default loop: multi-turn (tool -> tool -> text) -----------------


@pytest.mark.asyncio
async def test_agent_multi_turn() -> None:
    """LLM calls a tool, then calls another tool, then returns text."""
    my_agent = ai.agent(model=MOCK_MODEL, tools=[double, concat])

    turn1 = [
        tool_call_msg(tc_id="tc-1", name="concat", args='{"a": "hello", "b": " world"}')
    ]
    turn2 = [tool_call_msg(tc_id="tc-2", name="double", args='{"x": 3}', id="msg-2")]
    turn3 = [text_msg("Done: hello world, 6", id="msg-3")]
    llm = mock_llm([turn1, turn2, turn3])

    result = my_agent.run(ai.make_messages(user="Concat then double"))
    [m async for m in result]
    assert llm.call_count == 3


# -- execute_tool: missing tool raises ------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_missing_raises() -> None:
    """execute_tool with unknown tool name raises ValueError.

    Wrapped in ExceptionGroup by TaskGroup.
    """
    tc = messages.ToolCallPart(
        tool_call_id="tc-1", tool_name="nonexistent_tool_zzz", tool_args="{}"
    )
    my_agent = ai.agent(model=MOCK_MODEL, tools=[])

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> None:
        await ai.execute_tool(tc)

    mock_llm([])
    result = my_agent.run(ai.make_messages(user="go"))
    with pytest.raises(ExceptionGroup) as exc_info:
        [m async for m in result]
    assert any(isinstance(e, ValueError) for e in exc_info.value.exceptions)


# -- execute_tool: Runtime injection ---------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_injects_runtime() -> None:
    """Tools with a Runtime parameter get the active runtime injected."""
    received_rt = None

    @ai.tool
    async def introspect(query: str, rt: Runtime) -> str:
        """Tool that inspects runtime."""
        nonlocal received_rt
        received_rt = rt
        return "ok"

    my_agent = ai.agent(model=MOCK_MODEL, tools=[introspect])

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> None:
        result = await ai.stream_step(agent.model, msgs, agent.tools)
        if result.tool_calls:
            await asyncio.gather(
                *(
                    ai.execute_tool(tc, message=result.last_message)
                    for tc in result.tool_calls
                )
            )

    call = [tool_call_msg(tc_id="tc-1", name="introspect", args='{"query": "test"}')]
    mock_llm([call])
    result = my_agent.run(ai.make_messages(user="go"))
    [m async for m in result]
    assert received_rt is not None
    assert isinstance(received_rt, Runtime)


# -- execute_tool: returns updated ToolPart --------------------------------


@pytest.mark.asyncio
async def test_execute_tool_returns_result_part() -> None:
    """execute_tool returns a ToolResultPart; the original ToolCallPart is unchanged."""
    my_agent = ai.agent(model=MOCK_MODEL, tools=[double])

    @my_agent.loop
    async def custom(agent: ai.Agent, msgs: list[ai.Message]) -> None:
        result = await ai.stream_step(agent.model, msgs, agent.tools)
        if result.tool_calls:
            msg = result.last_message
            assert msg is not None
            for tc in result.tool_calls:
                result_part = await ai.execute_tool(tc, message=msg)
                # Returned part is a ToolResultPart with the result
                assert not result_part.is_error
                assert result_part.result == 10
            # Original message's tool calls are unchanged (immutable)
            assert msg.tool_calls[0].tool_name == "double"

    call = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 5}')]
    mock_llm([call])
    result = my_agent.run(ai.make_messages(user="go"))
    [m async for m in result]


# -- Checkpoint records tools from Agent default loop ----------------------


@pytest.mark.asyncio
async def test_agent_checkpoint_records_tools() -> None:
    """Agent default loop's tool executions are recorded in the checkpoint."""
    my_agent = ai.agent(model=MOCK_MODEL, tools=[double])

    call1 = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 4}')]
    call2 = [text_msg("8", id="msg-2")]
    mock_llm([call1, call2])

    result = my_agent.run(ai.make_messages(user="Double 4"))
    [m async for m in result]

    cp = result.checkpoint
    assert any(t.tool_call_id == "tc-1" and t.result == 8 for t in cp.tools)
