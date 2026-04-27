"""Agent default loop, tool execution, multi-turn."""

from __future__ import annotations

import ai
from ai.types import messages

from ..conftest import MOCK_MODEL, collect_messages, mock_llm, text_msg, tool_call_msg

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


async def test_agent_text_only() -> None:
    """Agent default loop with no tool calls returns after one LLM call."""
    my_agent = ai.agent(tools=[double])

    llm = mock_llm([[text_msg("Hello!")]])
    msgs = await collect_messages(my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]))
    assert llm.call_count == 1
    assert any(m.text == "Hello!" for m in msgs)


# -- Agent default loop: tool call + follow-up -----------------------------


async def test_agent_tool_then_text() -> None:
    """Agent default loop calls tool, feeds result back, gets final text."""
    my_agent = ai.agent(tools=[double])

    call1 = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 5}')]
    call2 = [text_msg("The answer is 10.")]
    llm = mock_llm([call1, call2])

    msgs = await collect_messages(
        my_agent.run(MOCK_MODEL, [ai.user_message("Double 5")])
    )
    assert llm.call_count == 2
    tool_results = [m for m in msgs if m.role == "tool" and m.tool_results]
    assert len(tool_results) >= 1
    assert tool_results[0].tool_results[0].result == 10


# -- Agent default loop: multiple tool calls in one message ----------------


async def test_agent_parallel_tools() -> None:
    """LLM returns two tool calls in one message; both execute."""
    my_agent = ai.agent(tools=[double])

    two_tools = messages.Message(
        id="msg-1",
        role="assistant",
        parts=[
            messages.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="double",
                tool_args='{"x": 3}',
            ),
            messages.ToolCallPart(
                tool_call_id="tc-2",
                tool_name="double",
                tool_args='{"x": 7}',
            ),
        ],
    )
    call2 = [text_msg("6 and 14", id="msg-2")]
    llm = mock_llm([[two_tools], call2])

    msgs = await collect_messages(
        my_agent.run(MOCK_MODEL, [ai.user_message("Double 3 and 7")])
    )
    assert llm.call_count == 2
    tool_result_msgs = [m for m in msgs if m.role == "tool" and m.tool_results]
    assert len(tool_result_msgs) >= 1


# -- Agent default loop: multi-turn (tool -> tool -> text) -----------------


async def test_agent_multi_turn() -> None:
    """LLM calls a tool, then calls another tool, then returns text."""
    my_agent = ai.agent(tools=[double, concat])

    turn1 = [
        tool_call_msg(tc_id="tc-1", name="concat", args='{"a": "hello", "b": " world"}')
    ]
    turn2 = [tool_call_msg(tc_id="tc-2", name="double", args='{"x": 3}', id="msg-2")]
    turn3 = [text_msg("Done: hello world, 6", id="msg-3")]
    llm = mock_llm([turn1, turn2, turn3])

    await collect_messages(
        my_agent.run(MOCK_MODEL, [ai.user_message("Concat then double")])
    )
    assert llm.call_count == 3


# -- turn_id semantics: one turn per LLM round-trip -------------------------


async def test_two_user_messages_produce_four_turns() -> None:
    """Two agent.run invocations, each with a tool call + final reply,
    produce four distinct turn ids; history from the first run keeps its
    original turn ids when fed into the second run."""
    my_agent = ai.agent(tools=[double])

    # Run 1: tool call, then text.
    r1_turn1 = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 5}', id="m-1a")]
    r1_turn2 = [text_msg("Ten.", id="m-1b")]
    # Run 2: tool call, then text.
    r2_turn1 = [tool_call_msg(tc_id="tc-2", name="double", args='{"x": 7}', id="m-2a")]
    r2_turn2 = [text_msg("Fourteen.", id="m-2b")]
    mock_llm([r1_turn1, r1_turn2, r2_turn1, r2_turn2])

    def dedup(stream: list[ai.Message]) -> list[ai.Message]:
        seen: dict[str, ai.Message] = {}
        for m in stream:
            seen[m.id] = m
        return list(seen.values())

    run1_stream = await collect_messages(
        my_agent.run(MOCK_MODEL, [ai.user_message("Double 5")])
    )
    history = dedup(run1_stream)

    run2_stream = await collect_messages(
        my_agent.run(MOCK_MODEL, [*history, ai.user_message("Double 7")])
    )
    final = dedup(run2_stream)

    # Chronological list of terminal non-internal messages.  Insertion order
    # of ``dedup`` reflects the order they first appeared in the stream.
    chronological = [m for m in final if m.role != "internal"]
    assert len(chronological) == 8

    # Expected shape: four turns, each a (input, assistant) pair.
    # turn 1: user → assistant (tool call)
    # turn 2: tool → assistant (text)
    # turn 3: user → assistant (tool call)
    # turn 4: tool → assistant (text)
    expected_roles = [
        ("user", "assistant"),
        ("tool", "assistant"),
        ("user", "assistant"),
        ("tool", "assistant"),
    ]
    pairs = [(chronological[2 * i], chronological[2 * i + 1]) for i in range(4)]
    for (left, right), (expected_left, expected_right) in zip(
        pairs, expected_roles, strict=True
    ):
        assert (left.role, right.role) == (expected_left, expected_right)
        # Both messages in the pair share the same turn_id.
        assert left.turn_id is not None
        assert left.turn_id == right.turn_id

    # The four turn_ids are all distinct.
    turn_ids = [left.turn_id for left, _ in pairs]
    assert len(set(turn_ids)) == 4

    # Run 1's history survives untouched into run 2.
    history_ids = {h.id for h in history}
    for h in history:
        same = next(m for m in final if m.id == h.id)
        assert same.turn_id == h.turn_id
    assert any(m.id in history_ids for m in final)
