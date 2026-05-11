"""Tests for Google stream parsing."""

from __future__ import annotations

import json

import pytest
from google.genai import types as genai_types

import ai
from ai import models
from ai.models.google import adapter, google

from .conftest import FakeGoogleClient, FakeGoogleStream

_CLIENT = models.Client(base_url="https://google.test/v1beta", api_key="sk-test")
_MODEL = google("gemini-2.5-flash")


def _chunk(
    *parts: dict[str, object],
    finish_reason: str | None = None,
) -> genai_types.GenerateContentResponse:
    candidate: dict[str, object] = {
        "content": {"role": "model", "parts": list(parts)},
    }
    if finish_reason is not None:
        candidate.update(
            {
                "finish_reason": finish_reason,
                "safety_ratings": [{"category": "HARM_CATEGORY_HATE_SPEECH"}],
            }
        )

    payload: dict[str, object] = {"candidates": [candidate]}
    if finish_reason is not None:
        payload["usage_metadata"] = {
            "prompt_token_count": 5,
            "candidates_token_count": 7,
            "thoughts_token_count": 2,
            "cached_content_token_count": 1,
        }
    return genai_types.GenerateContentResponse.model_validate(payload)


def _part(**values: object) -> dict[str, object]:
    return genai_types.Part.model_validate(values).model_dump(
        mode="json",
        exclude_none=True,
    )


async def _drain(
    stream: FakeGoogleStream,
    monkeypatch: pytest.MonkeyPatch,
) -> models.Stream:
    fake = FakeGoogleClient(stream=stream)
    monkeypatch.setattr(adapter, "_make_client", lambda client: fake)
    s = models.Stream(adapter.stream(_CLIENT, _MODEL, [ai.user_message("Hi")]))
    async for _ in s:
        pass
    return s


async def test_text_reasoning_file_and_function_call_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_stream = FakeGoogleStream(
        [
            _chunk(_part(text="hello")),
            _chunk(_part(text="thinking", thought=True)),
            _chunk(_part(inline_data={"mime_type": "image/png", "data": "cG5n"})),
            _chunk(
                _part(
                    function_call={
                        "id": "fc_1",
                        "name": "weather",
                        "args": {"city": "SF"},
                    }
                ),
                finish_reason="STOP",
            ),
        ]
    )

    s = await _drain(sdk_stream, monkeypatch)

    assert s.message.text == "hello"
    assert s.message.reasoning == "thinking"
    assert s.message.usage is not None
    assert s.message.usage.input_tokens == 5
    assert s.message.usage.output_tokens == 9
    assert s.message.usage.reasoning_tokens == 2
    assert s.message.usage.cache_read_tokens == 1
    assert s.message.provider_metadata is not None
    assert s.message.provider_metadata["provider"] == "google"
    assert s.message.provider_metadata["finishReason"] == "STOP"

    file = s.message.files[0]
    assert file.media_type == "image/png"
    assert file.data == "cG5n"

    calls = s.message.tool_calls
    assert len(calls) == 1
    assert calls[0].tool_call_id == "fc_1"
    assert calls[0].tool_name == "weather"
    assert calls[0].tool_args == '{"city":"SF"}'


async def test_code_execution_emits_builtin_call_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_stream = FakeGoogleStream(
        [
            _chunk(
                _part(
                    executable_code={
                        "id": "code_1",
                        "language": "PYTHON",
                        "code": "print(1)",
                    }
                )
            ),
            _chunk(
                _part(
                    code_execution_result={
                        "id": "code_1",
                        "outcome": "OUTCOME_OK",
                        "output": "1\n",
                    }
                )
            ),
        ]
    )

    s = await _drain(sdk_stream, monkeypatch)

    calls = s.message.builtin_tool_calls
    assert len(calls) == 1
    assert calls[0].tool_call_id == "code_1"
    assert calls[0].tool_name == "code_execution"
    assert json.loads(calls[0].tool_args) == {
        "id": "code_1",
        "language": "PYTHON",
        "code": "print(1)",
    }

    returns = s.message.builtin_tool_returns
    assert len(returns) == 1
    assert returns[0].tool_call_id == "code_1"
    assert returns[0].tool_name == "code_execution"
    assert returns[0].result == {
        "id": "code_1",
        "outcome": "OUTCOME_OK",
        "output": "1\n",
    }


async def test_server_tool_call_and_response_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_stream = FakeGoogleStream(
        [
            _chunk(
                _part(
                    tool_call={
                        "id": "srv_1",
                        "tool_type": "GOOGLE_SEARCH_WEB",
                        "args": {"query": "weather"},
                    }
                )
            ),
            _chunk(
                _part(
                    tool_response={
                        "id": "srv_1",
                        "tool_type": "GOOGLE_SEARCH_WEB",
                        "response": {"results": [{"title": "Forecast"}]},
                    }
                )
            ),
        ]
    )

    s = await _drain(sdk_stream, monkeypatch)

    calls = s.message.builtin_tool_calls
    assert len(calls) == 1
    assert calls[0].tool_call_id == "srv_1"
    assert calls[0].tool_name == "server:GOOGLE_SEARCH_WEB"
    assert calls[0].tool_args == '{"query":"weather"}'
    assert calls[0].provider_metadata == {
        "provider": "google",
        "serverToolCallId": "srv_1",
        "serverToolType": "GOOGLE_SEARCH_WEB",
    }

    returns = s.message.builtin_tool_returns
    assert len(returns) == 1
    assert returns[0].tool_call_id == "srv_1"
    assert returns[0].tool_name == "server:GOOGLE_SEARCH_WEB"
    assert returns[0].result == {"results": [{"title": "Forecast"}]}


async def test_thought_signature_round_trips_from_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake = FakeGoogleClient(captured)
    monkeypatch.setattr(adapter, "_make_client", lambda client: fake)

    stream = adapter.stream(
        _CLIENT,
        _MODEL,
        [
            ai.assistant_message(
                ai.thinking(
                    "hidden",
                    provider_metadata={
                        "provider": "google",
                        "thoughtSignature": "sig",
                    },
                )
            )
        ],
    )
    async for _ in stream:
        pass

    assert captured["contents"] == [
        {
            "role": "model",
            "parts": [
                {
                    "text": "hidden",
                    "thought": True,
                    "thought_signature": "sig",
                }
            ],
        }
    ]
