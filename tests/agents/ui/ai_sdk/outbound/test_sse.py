from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from ai import events as agent_events_
from ai.agents.ui.ai_sdk import protocol, to_sse
from ai.agents.ui.ai_sdk.outbound.sse import format_sse, serialize_part
from ai.types import events as events_


def test_serialize_part_camelcases_keys() -> None:
    part = protocol.StartPart(message_id="m1")
    payload = json.loads(serialize_part(part))
    assert payload == {"type": "start", "messageId": "m1"}


def test_format_sse_wraps_data_line() -> None:
    part = protocol.TextDeltaPart(id="t1", delta="hi")
    line = format_sse(part)
    assert line.startswith("data: ")
    assert line.endswith("\n\n")


def test_serialize_data_part_uses_type_with_prefix() -> None:
    part = protocol.DataPart(data_type="custom", data={"k": 1})
    payload = json.loads(serialize_part(part))
    assert payload["type"] == "data-custom"
    assert "dataType" not in payload


async def _gen(
    stream_events: list[agent_events_.AgentEvent],
) -> AsyncGenerator[agent_events_.AgentEvent]:
    for event in stream_events:
        yield event


async def test_to_sse_emits_data_prefixed_lines() -> None:
    lines = [
        line
        async for line in to_sse(
            _gen(
                [
                    events_.TextStart(block_id="t1"),
                    events_.TextDelta(block_id="t1", chunk="hi"),
                    events_.TextEnd(block_id="t1"),
                ]
            )
        )
    ]
    assert all(line.startswith("data: ") for line in lines)
    # first line is the start part (lazy open)
    first = json.loads(lines[0].removeprefix("data: ").rstrip())
    assert first["type"] == "start"
