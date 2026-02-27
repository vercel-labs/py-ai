"""Message model: properties, ToolPart.set_result/set_error, make_messages,
StructuredOutputPart."""

import pydantic
import pytest

from vercel_ai_sdk.core.messages import (
    HookPart,
    Message,
    ReasoningPart,
    StructuredOutputPart,
    TextPart,
    ToolPart,
    Usage,
    make_messages,
)


class _Weather(pydantic.BaseModel):
    city: str
    temperature: float


_WEATHER_DATA = {"city": "SF", "temperature": 62.0}
_WEATHER_TYPE_NAME = f"{_Weather.__module__}.{_Weather.__qualname__}"

# -- is_done ---------------------------------------------------------------


def test_is_done_all_done() -> None:
    m = Message(
        id="m1",
        role="assistant",
        parts=[
            TextPart(text="hello", state="done"),
            ToolPart(tool_call_id="tc1", tool_name="t", tool_args="{}", state="done"),
        ],
    )
    assert m.is_done is True


def test_is_done_streaming() -> None:
    m = Message(
        id="m1",
        role="assistant",
        parts=[TextPart(text="hel", state="streaming", delta="hel")],
    )
    assert m.is_done is False


def test_is_done_no_state() -> None:
    """Parts without state (restored from storage) count as done."""
    m = Message(id="m1", role="assistant", parts=[TextPart(text="hi")])
    assert m.is_done is True


# -- text / reasoning properties -------------------------------------------


def test_text_returns_first_text_part() -> None:
    m = Message(
        id="m1",
        role="assistant",
        parts=[
            TextPart(text="first"),
            TextPart(text="second"),
        ],
    )
    assert m.text == "first"


def test_text_empty_when_no_text_parts() -> None:
    m = Message(
        id="m1",
        role="assistant",
        parts=[ToolPart(tool_call_id="tc1", tool_name="t", tool_args="{}")],
    )
    assert m.text == ""


def test_reasoning_returns_first() -> None:
    m = Message(
        id="m1",
        role="assistant",
        parts=[ReasoningPart(text="thinking hard"), TextPart(text="answer")],
    )
    assert m.reasoning == "thinking hard"


# -- deltas ----------------------------------------------------------------


def test_text_delta() -> None:
    m = Message(
        id="m1",
        role="assistant",
        parts=[TextPart(text="ab", delta="b", state="streaming")],
    )
    assert m.text_delta == "b"


def test_text_delta_empty_when_no_delta() -> None:
    m = Message(id="m1", role="assistant", parts=[TextPart(text="done", state="done")])
    assert m.text_delta == ""


def test_reasoning_delta() -> None:
    m = Message(
        id="m1",
        role="assistant",
        parts=[ReasoningPart(text="ab", delta="b", state="streaming")],
    )
    assert m.reasoning_delta == "b"


def test_tool_deltas() -> None:
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


def test_tool_calls() -> None:
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


def test_get_tool_part_found() -> None:
    m = Message(
        id="m1",
        role="assistant",
        parts=[ToolPart(tool_call_id="tc1", tool_name="t", tool_args="{}")],
    )
    tp = m.get_tool_part("tc1")
    assert tp is not None
    assert tp.tool_name == "t"


def test_get_tool_part_missing() -> None:
    m = Message(id="m1", role="assistant", parts=[TextPart(text="no tools")])
    assert m.get_tool_part("tc-nope") is None


# -- get_hook_part ---------------------------------------------------------


def test_get_hook_part_found() -> None:
    """get_hook_part returns the HookPart when present."""
    hook = HookPart(hook_id="h1", hook_type="Approval", status="pending")
    m = Message(id="m1", role="assistant", parts=[hook])
    assert m.get_hook_part() is hook
    assert m.get_hook_part("h1") is hook


def test_get_hook_part_by_id() -> None:
    """get_hook_part with a specific hook_id skips non-matching hooks."""
    h1 = HookPart(hook_id="h1", hook_type="Approval", status="pending")
    h2 = HookPart(hook_id="h2", hook_type="Approval", status="resolved")
    m = Message(id="m1", role="assistant", parts=[h1, h2])
    assert m.get_hook_part("h2") is h2


def test_get_hook_part_missing() -> None:
    """get_hook_part returns None when no HookPart exists."""
    m = Message(id="m1", role="assistant", parts=[TextPart(text="no hooks")])
    assert m.get_hook_part() is None
    assert m.get_hook_part("h-nope") is None


# -- ToolPart.set_result / set_error ---------------------------------------


def test_set_result() -> None:
    tp = ToolPart(tool_call_id="tc1", tool_name="t", tool_args="{}")
    assert tp.status == "pending"
    tp.set_result({"answer": 42})
    # mypy narrows status to Literal["pending"] from the constructor default and
    # can't track that set_result() mutates it to "result"
    assert tp.status == "result"  # type: ignore[comparison-overlap]
    assert tp.result == {"answer": 42}


def test_set_error() -> None:
    tp = ToolPart(tool_call_id="tc1", tool_name="t", tool_args="{}")
    assert tp.status == "pending"
    tp.set_error("Something went wrong")
    assert tp.status == "error"  # type: ignore[comparison-overlap]
    assert tp.result == "Something went wrong"


# -- make_messages ---------------------------------------------------------


def test_make_messages_system_and_user() -> None:
    msgs = make_messages(system="You are helpful.", user="Hi")
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[0].text == "You are helpful."
    assert msgs[1].role == "user"
    assert msgs[1].text == "Hi"


def test_make_messages_user_only() -> None:
    msgs = make_messages(user="Hi")
    assert len(msgs) == 1
    assert msgs[0].role == "user"


# -- StructuredOutputPart --------------------------------------------------


def test_structured_output_part_value() -> None:
    """Lazy hydration: resolves class, validates data, caches result."""
    part = StructuredOutputPart(data=_WEATHER_DATA, output_type_name=_WEATHER_TYPE_NAME)
    val = part.value
    assert isinstance(val, _Weather)
    assert val.city == "SF"
    assert part.value is val  # cached


def test_structured_output_part_bad_class_name() -> None:
    """Unresolvable class name raises ImportError on access."""
    part = StructuredOutputPart(
        data=_WEATHER_DATA, output_type_name="nonexistent.module.Cls"
    )
    with pytest.raises(ImportError):
        _ = part.value


def test_message_output_from_part() -> None:
    """Message.output property delegates to StructuredOutputPart.value."""
    m = Message(
        id="m1",
        role="assistant",
        parts=[
            TextPart(text="{}"),
            StructuredOutputPart(
                data=_WEATHER_DATA, output_type_name=_WEATHER_TYPE_NAME
            ),
        ],
    )
    assert isinstance(m.output, _Weather)
    assert m.output.city == "SF"


def test_structured_output_round_trip() -> None:
    """StructuredOutputPart survives model_dump -> model_validate."""
    m = Message(
        id="m1",
        role="assistant",
        parts=[
            TextPart(text="{}"),
            StructuredOutputPart(
                data=_WEATHER_DATA, output_type_name=_WEATHER_TYPE_NAME
            ),
        ],
    )
    restored = Message.model_validate(m.model_dump())
    assert isinstance(restored.output, _Weather)
    assert restored.output.city == "SF"


# -- Usage -----------------------------------------------------------------


def test_usage_add_merges_optional_fields() -> None:
    """__add__ accumulates tokens and treats None vs populated correctly."""
    a = Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=20,
        # reasoning_tokens and cache_write_tokens left as None
    )
    b = Usage(
        input_tokens=200,
        output_tokens=80,
        reasoning_tokens=10,
        # cache_read_tokens left as None, cache_write_tokens left as None
    )
    total = a + b

    assert total.input_tokens == 300
    assert total.output_tokens == 130
    assert total.total_tokens == 430

    # None + int -> int (not None)
    assert total.reasoning_tokens == 10
    # int + None -> int (not None)
    assert total.cache_read_tokens == 20
    # None + None -> None (not zero)
    assert total.cache_write_tokens is None

    # raw is intentionally not merged
    assert total.raw is None
