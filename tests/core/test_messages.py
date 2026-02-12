"""Message model: properties, ToolPart.set_result, make_messages."""

from vercel_ai_sdk.core.messages import (
    Message,
    ReasoningPart,
    TextPart,
    ToolDelta,
    ToolPart,
    make_messages,
)


# -- is_done ---------------------------------------------------------------


def test_is_done_all_done():
    m = Message(
        id="m1",
        role="assistant",
        parts=[
            TextPart(text="hello", state="done"),
            ToolPart(tool_call_id="tc1", tool_name="t", tool_args="{}", state="done"),
        ],
    )
    assert m.is_done is True


def test_is_done_streaming():
    m = Message(
        id="m1",
        role="assistant",
        parts=[TextPart(text="hel", state="streaming", delta="hel")],
    )
    assert m.is_done is False


def test_is_done_no_state():
    """Parts without state (restored from storage) count as done."""
    m = Message(id="m1", role="assistant", parts=[TextPart(text="hi")])
    assert m.is_done is True


# -- text / reasoning properties -------------------------------------------


def test_text_returns_first_text_part():
    m = Message(
        id="m1",
        role="assistant",
        parts=[
            TextPart(text="first"),
            TextPart(text="second"),
        ],
    )
    assert m.text == "first"


def test_text_empty_when_no_text_parts():
    m = Message(
        id="m1",
        role="assistant",
        parts=[ToolPart(tool_call_id="tc1", tool_name="t", tool_args="{}")],
    )
    assert m.text == ""


def test_reasoning_returns_first():
    m = Message(
        id="m1",
        role="assistant",
        parts=[ReasoningPart(text="thinking hard"), TextPart(text="answer")],
    )
    assert m.reasoning == "thinking hard"


# -- deltas ----------------------------------------------------------------


def test_text_delta():
    m = Message(
        id="m1",
        role="assistant",
        parts=[TextPart(text="ab", delta="b", state="streaming")],
    )
    assert m.text_delta == "b"


def test_text_delta_empty_when_no_delta():
    m = Message(id="m1", role="assistant", parts=[TextPart(text="done", state="done")])
    assert m.text_delta == ""


def test_reasoning_delta():
    m = Message(
        id="m1",
        role="assistant",
        parts=[ReasoningPart(text="ab", delta="b", state="streaming")],
    )
    assert m.reasoning_delta == "b"


def test_tool_deltas():
    m = Message(
        id="m1",
        role="assistant",
        parts=[
            ToolPart(
                tool_call_id="tc1",
                tool_name="search",
                tool_args='{"q":"te',
                args_delta='"te',
                state="streaming",
            )
        ],
    )
    deltas = m.tool_deltas
    assert len(deltas) == 1
    assert deltas[0].tool_call_id == "tc1"
    assert deltas[0].args_delta == '"te'


# -- tool_calls / get_tool_part -------------------------------------------


def test_tool_calls():
    m = Message(
        id="m1",
        role="assistant",
        parts=[
            TextPart(text="hi"),
            ToolPart(tool_call_id="tc1", tool_name="a", tool_args="{}"),
            ToolPart(tool_call_id="tc2", tool_name="b", tool_args="{}"),
        ],
    )
    assert len(m.tool_calls) == 2
    assert m.tool_calls[0].tool_call_id == "tc1"


def test_get_tool_part_found():
    m = Message(
        id="m1",
        role="assistant",
        parts=[ToolPart(tool_call_id="tc1", tool_name="t", tool_args="{}")],
    )
    assert m.get_tool_part("tc1") is not None
    assert m.get_tool_part("tc1").tool_name == "t"


def test_get_tool_part_missing():
    m = Message(id="m1", role="assistant", parts=[TextPart(text="no tools")])
    assert m.get_tool_part("tc-nope") is None


# -- ToolPart.set_result ---------------------------------------------------


def test_set_result():
    tp = ToolPart(tool_call_id="tc1", tool_name="t", tool_args="{}")
    assert tp.status == "pending"
    tp.set_result({"answer": 42})
    assert tp.status == "result"
    assert tp.result == {"answer": 42}


# -- make_messages ---------------------------------------------------------


def test_make_messages_system_and_user():
    msgs = make_messages(system="You are helpful.", user="Hi")
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[0].text == "You are helpful."
    assert msgs[1].role == "user"
    assert msgs[1].text == "Hi"


def test_make_messages_user_only():
    msgs = make_messages(user="Hi")
    assert len(msgs) == 1
    assert msgs[0].role == "user"
