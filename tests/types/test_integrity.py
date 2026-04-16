"""Tests for message integrity checker."""

from __future__ import annotations

from typing import Literal

import pytest

from ai.types import builders, messages
from ai.types.integrity import IntegrityError, prepare_messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant_with_tool_call(
    tool_call_id: str = "tc-1",
    tool_name: str = "calc",
    tool_args: str = '{"x": 1}',
) -> messages.Message:
    return messages.Message(
        role="assistant",
        parts=[
            messages.ToolCallPart(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_args=tool_args,
            )
        ],
    )


def _tool_result(
    tool_call_id: str = "tc-1",
    tool_name: str = "calc",
    result: str = "42",
) -> messages.Message:
    return builders.tool_message(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=result,
    )


def _assert_raises_issue(
    msgs: list[messages.Message],
    issue: str,
    *,
    mode: Literal["auto", "strict"] = "auto",
) -> None:
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs, mode=mode)
    assert issue in exc_info.value.issues


# ---------------------------------------------------------------------------
# Clean passthrough
# ---------------------------------------------------------------------------


def test_clean_messages_pass_through() -> None:
    msgs = [
        builders.user_message("hello"),
        builders.assistant_message("world"),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 2
    assert result[0].text == "hello"
    assert result[1].text == "world"


def test_idempotent() -> None:
    msgs = [
        builders.user_message("hi"),
        _assistant_with_tool_call(),
        _tool_result(),
        builders.assistant_message("done"),
    ]
    once = prepare_messages(msgs)
    twice = prepare_messages(once)
    assert len(once) == len(twice)
    for a, b in zip(once, twice, strict=True):
        assert a.role == b.role
        assert len(a.parts) == len(b.parts)


# ---------------------------------------------------------------------------
# Signal messages
# ---------------------------------------------------------------------------


def test_drops_signal_messages() -> None:
    msgs = [
        builders.user_message("hi"),
        messages.Message(role="signal", parts=[messages.TextPart(text="internal")]),
        builders.assistant_message("hello"),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"


def test_signal_strict_raises() -> None:
    msgs = [
        messages.Message(role="signal", parts=[messages.TextPart(text="x")]),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs, mode="strict")
    assert "signal-message" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Internal parts (HookPart, StructuredOutputPart)
# ---------------------------------------------------------------------------


def test_strips_internal_parts() -> None:
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.TextPart(text="hi"),
            messages.HookPart(hook_id="h1", hook_type="confirm", status="resolved"),
        ],
    )
    result = prepare_messages([msg])
    assert len(result) == 1
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], messages.TextPart)


def test_strips_internal_parts_drops_empty_message() -> None:
    """Message with only internal parts becomes empty and is dropped."""
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.HookPart(hook_id="h1", hook_type="confirm", status="resolved"),
        ],
    )
    result = prepare_messages([msg])
    assert len(result) == 0


def test_internal_parts_strict_raises() -> None:
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.HookPart(hook_id="h1", hook_type="confirm", status="resolved"),
        ],
    )
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages([msg], mode="strict")
    assert "internal-part" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Invalid tool args
# ---------------------------------------------------------------------------


def test_fixes_invalid_tool_args() -> None:
    msg = _assistant_with_tool_call(tool_args="not json {{{")
    result = prepare_messages([msg])
    tc = result[0].parts[0]
    assert isinstance(tc, messages.ToolCallPart)
    assert tc.tool_args == "{}"


def test_preserves_valid_tool_args() -> None:
    msg = _assistant_with_tool_call(tool_args='{"key": "value"}')
    result = prepare_messages([msg])
    tc = result[0].parts[0]
    assert isinstance(tc, messages.ToolCallPart)
    assert tc.tool_args == '{"key": "value"}'


def test_invalid_tool_args_strict_raises() -> None:
    msg = _assistant_with_tool_call(tool_args="broken")
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages([msg], mode="strict")
    assert "invalid-tool-args" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Orphaned tool calls (no matching result) — auto-fixable
# ---------------------------------------------------------------------------


def test_inserts_synthetic_result_for_orphaned_call_at_end() -> None:
    """Tool call at end of history with no result gets a synthetic one."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 3
    assert result[2].role == "tool"
    tr = result[2].tool_results[0]
    assert tr.tool_call_id == "tc-1"
    assert tr.is_error is True


def test_inserts_synthetic_result_before_user_interruption() -> None:
    """User message interrupting tool flow triggers synthetic results."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
        builders.user_message("never mind"),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 4
    # Synthetic result inserted before the user message.
    assert result[2].role == "tool"
    assert result[2].tool_results[0].is_error is True
    assert result[3].role == "user"
    assert result[3].text == "never mind"


def test_inserts_synthetic_result_before_next_assistant() -> None:
    """New assistant message while tool calls pending triggers synthetic results."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
        builders.assistant_message("actually, the answer is 4"),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 4
    assert result[2].role == "tool"
    assert result[2].tool_results[0].is_error is True
    assert result[3].role == "assistant"


def test_multiple_orphaned_calls_get_individual_results() -> None:
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.ToolCallPart(tool_call_id="tc-1", tool_name="a", tool_args="{}"),
            messages.ToolCallPart(tool_call_id="tc-2", tool_name="b", tool_args="{}"),
        ],
    )
    result = prepare_messages([builders.user_message("go"), msg])
    # Synthetic tool message should have results for both calls.
    synthetic = result[2]
    assert synthetic.role == "tool"
    ids = {tr.tool_call_id for tr in synthetic.tool_results}
    assert ids == {"tc-1", "tc-2"}


def test_partial_results_only_fills_missing() -> None:
    """If some results exist, only the missing ones get synthetic fills."""
    msgs = [
        builders.user_message("go"),
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="tc-1", tool_name="a", tool_args="{}"
                ),
                messages.ToolCallPart(
                    tool_call_id="tc-2", tool_name="b", tool_args="{}"
                ),
            ],
        ),
        _tool_result(tool_call_id="tc-1"),
        # tc-2 is missing, then user interrupts
        builders.user_message("stop"),
    ]
    result = prepare_messages(msgs)
    # user, assistant, tool(tc-1), synthetic-tool(tc-2), user
    assert len(result) == 5
    synthetic = result[3]
    assert synthetic.role == "tool"
    assert len(synthetic.tool_results) == 1
    assert synthetic.tool_results[0].tool_call_id == "tc-2"
    assert synthetic.tool_results[0].is_error is True


def test_orphaned_tool_call_strict_raises() -> None:
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs, mode="strict")
    assert "orphaned-tool-call" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Orphaned tool results (no matching call) — always raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "strict"])
def test_orphaned_tool_result_always_raises(
    mode: Literal["auto", "strict"],
) -> None:
    """Tool result referencing a nonexistent call always raises."""
    msgs = [
        builders.user_message("hi"),
        _tool_result(tool_call_id="nonexistent"),
    ]
    _assert_raises_issue(msgs, "orphaned-tool-result", mode=mode)


def test_out_of_sequence_tool_result_raises() -> None:
    """A late tool result cannot arrive after another conversation turn."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(),
        builders.user_message("never mind"),
        _tool_result(),
    ]
    _assert_raises_issue(msgs, "orphaned-tool-result")


# ---------------------------------------------------------------------------
# Complete tool flow (no issues)
# ---------------------------------------------------------------------------


def test_complete_tool_flow_unchanged() -> None:
    """A properly paired tool flow passes through without modification."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
        _tool_result(),
        builders.assistant_message("The answer is 4"),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 4
    assert [m.role for m in result] == ["user", "assistant", "tool", "assistant"]


# ---------------------------------------------------------------------------
# Strict mode collects multiple issues
# ---------------------------------------------------------------------------


def test_strict_collects_all_issues() -> None:
    msgs = [
        messages.Message(role="signal", parts=[messages.TextPart(text="x")]),
        messages.Message(
            role="assistant",
            parts=[
                messages.TextPart(text="hi"),
                messages.HookPart(hook_id="h1", hook_type="confirm", status="resolved"),
            ],
        ),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs, mode="strict")
    issues = exc_info.value.issues
    assert "signal-message" in issues
    assert "internal-part" in issues


def test_strict_keeps_recoverable_issues_when_history_is_corrupt() -> None:
    msgs = [
        messages.Message(role="signal", parts=[messages.TextPart(text="x")]),
        builders.user_message("go"),
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="tc-1", tool_name="a", tool_args="{}"
                ),
                messages.ToolCallPart(
                    tool_call_id="tc-1", tool_name="b", tool_args="{}"
                ),
            ],
        ),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs, mode="strict")
    assert "signal-message" in exc_info.value.issues
    assert "duplicate-tool-call" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Duplicate tool call IDs — always raises
# ---------------------------------------------------------------------------


def test_duplicate_tool_calls_raises_in_auto() -> None:
    """Two assistant messages using the same tool_call_id always raises."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(tool_call_id="tc-1", tool_args='{"v": 1}'),
        _tool_result(tool_call_id="tc-1", result="old"),
        _assistant_with_tool_call(tool_call_id="tc-1", tool_args='{"v": 2}'),
        _tool_result(tool_call_id="tc-1", result="new"),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs)
    assert "duplicate-tool-call" in exc_info.value.issues
    assert "duplicate-tool-result" in exc_info.value.issues


def test_duplicate_tool_calls_within_same_message_raises() -> None:
    """Two tool calls with the same ID in one assistant message raises."""
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.ToolCallPart(
                tool_call_id="tc-1", tool_name="a", tool_args='{"v": 1}'
            ),
            messages.ToolCallPart(
                tool_call_id="tc-1", tool_name="a", tool_args='{"v": 2}'
            ),
        ],
    )
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages([builders.user_message("go"), msg])
    assert "duplicate-tool-call" in exc_info.value.issues


def test_duplicate_tool_results_raises_in_auto() -> None:
    """Two tool messages with results for the same call always raises."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(tool_call_id="tc-1"),
        _tool_result(tool_call_id="tc-1", result="first"),
        _tool_result(tool_call_id="tc-1", result="second"),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs)
    assert "duplicate-tool-result" in exc_info.value.issues


def test_duplicate_tool_results_within_same_message_raises() -> None:
    """Two results for the same call ID in one tool message raises."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(tool_call_id="tc-1"),
        messages.Message(
            role="tool",
            parts=[
                builders.tool_result("tc-1", result="first"),
                builders.tool_result("tc-1", result="second"),
            ],
        ),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs)
    assert "duplicate-tool-result" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Does not mutate input
# ---------------------------------------------------------------------------


def test_does_not_mutate_input() -> None:
    original = [
        builders.user_message("hi"),
        _assistant_with_tool_call(),
    ]
    original_len = len(original)
    _ = prepare_messages(original)
    assert len(original) == original_len
