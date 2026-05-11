"""Tests for the Google adapter's request shaping."""

from __future__ import annotations

from typing import Any

import pydantic
import pytest

import ai
from ai import models
from ai.models.google import adapter, google
from ai.models.google import tools as google_tools
from ai.types import messages
from ai.types import tools as tool_types

from .conftest import FakeGoogleClient


class _Answer(pydantic.BaseModel):
    answer: str


_TEST_CLIENT = models.Client(base_url="https://google.test/v1beta", api_key="sk-test")
_MODEL = google("gemini-2.5-flash")


def _patch(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    fake = FakeGoogleClient(captured)
    monkeypatch.setattr(adapter, "_make_client", lambda client: fake)
    return captured


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


async def test_system_and_user_content_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [
                ai.system_message("rules"),
                ai.user_message(
                    "Describe this",
                    messages.FilePart(
                        data=b"img",
                        media_type="image/png",
                        filename="image.png",
                    ),
                ),
            ],
        )
    )

    assert captured["model"] == "gemini-2.5-flash"
    assert captured["sdk_contents"]
    assert captured["sdk_config"] is not None
    assert captured["config"]["system_instruction"] == "rules"
    assert captured["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "Describe this"},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": "aW1n",
                        "display_name": "image.png",
                    }
                },
            ],
        }
    ]


async def test_raw_params_pass_through_to_generate_content_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params={
                "temperature": 0.2,
                "top_p": 0.9,
                "thinking_config": {"thinking_budget": 128},
            },
        )
    )

    assert captured["config"]["temperature"] == 0.2
    assert captured["config"]["top_p"] == 0.9
    assert captured["config"]["thinking_config"] == {"thinking_budget": 128}


async def test_structured_output_maps_to_json_schema_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            output_type=_Answer,
        )
    )

    assert captured["config"]["response_mime_type"] == "application/json"
    assert captured["config"]["response_json_schema"]["title"] == "_Answer"


async def test_function_and_provider_tools_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)
    weather = tool_types.Tool(
        kind="function",
        name="weather",
        args=tool_types.FunctionToolArgs(
            description="Get weather",
            params={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    )

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            tools=[
                weather,
                google_tools.google_search(
                    search_types={"web_search": {}},
                    time_range_filter={
                        "start_time": "2026-01-01T00:00:00Z",
                        "end_time": "2026-01-02T00:00:00Z",
                    },
                ),
                google_tools.file_search(
                    file_search_store_names=["fileSearchStores/store-1"],
                    top_k=3,
                ),
            ],
        )
    )

    assert captured["config"]["tools"] == [
        {
            "function_declarations": [
                {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters_json_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        },
        {
            "google_search": {
                "search_types": {"web_search": {}},
                "time_range_filter": {
                    "start_time": "2026-01-01T00:00:00Z",
                    "end_time": "2026-01-02T00:00:00Z",
                },
            }
        },
        {
            "file_search": {
                "file_search_store_names": ["fileSearchStores/store-1"],
                "top_k": 3,
            }
        },
    ]
    assert captured["config"]["tool_config"] == {
        "include_server_side_tool_invocations": True
    }
    assert captured["sdk_config"] is not None


async def test_non_dict_params_rejected_by_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)

    stream = adapter.stream(
        _TEST_CLIENT,
        _MODEL,
        [ai.user_message("Hi")],
        params=[{"temperature": 0.2}],
    )

    with pytest.raises(TypeError, match="dict"):
        await _drain(stream)


async def test_message_history_round_trips_tool_calls_and_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)
    convo = [
        ai.user_message("Weather?"),
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="fc_1",
                    tool_name="weather",
                    tool_args='{"city":"SF"}',
                )
            ],
        ),
        ai.tool_message(
            tool_call_id="fc_1",
            tool_name="weather",
            result={"temp": 62},
        ),
    ]

    await _drain(adapter.stream(_TEST_CLIENT, _MODEL, convo))

    assert captured["contents"][1] == {
        "role": "model",
        "parts": [
            {
                "function_call": {
                    "id": "fc_1",
                    "name": "weather",
                    "args": {"city": "SF"},
                }
            }
        ],
    }
    assert captured["contents"][2] == {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "id": "fc_1",
                    "name": "weather",
                    "response": {"output": {"temp": 62}},
                }
            }
        ],
    }
