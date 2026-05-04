from __future__ import annotations

from typing import Any

import pydantic
import pytest

import ai
from ai import models
from ai.models.openai import adapter, openai
from ai.models.openai import params as openai_params


class _Answer(pydantic.BaseModel):
    answer: str


class _EmptyOpenAIStream:
    def __aiter__(self) -> _EmptyOpenAIStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class _FakeCompletions:
    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    async def create(self, **kwargs: Any) -> _EmptyOpenAIStream:
        self._captured.update(kwargs)
        return _EmptyOpenAIStream()


class _FakeChat:
    def __init__(self, captured: dict[str, Any]) -> None:
        self.completions = _FakeCompletions(captured)


class _FakeOpenAIClient:
    def __init__(self, captured: dict[str, Any]) -> None:
        self.chat = _FakeChat(captured)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_chat_params_map_to_request_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_client = _FakeOpenAIClient(captured)
    monkeypatch.setattr(adapter, "_make_client", lambda client: fake_client)

    stream = adapter.stream(
        models.Client(base_url="https://openai.test", api_key="sk-test"),
        openai("gpt-5.4"),
        [ai.system_message("developer rules"), ai.user_message("Hi")],
        output_type=_Answer,
        params=openai_params.OpenAIChatParams(
            reasoning_effort="high",
            service_tier="priority",
            parallel_tool_calls=False,
            logprobs=3,
            strict_json_schema=False,
            system_message_mode="developer",
            text_verbosity="low",
            extra_body={"future_option": {"enabled": True}},
            extra_headers={"x-openai-feature": "enabled"},
        ),
    )
    events = [event async for event in stream]

    assert [type(event).__name__ for event in events] == ["StreamStart", "StreamEnd"]
    assert captured["messages"][0] == {
        "role": "developer",
        "content": "developer rules",
    }
    assert captured["reasoning_effort"] == "high"
    assert captured["service_tier"] == "priority"
    assert captured["parallel_tool_calls"] is False
    assert captured["logprobs"] is True
    assert captured["top_logprobs"] == 3
    assert captured["verbosity"] == "low"
    assert captured["response_format"]["json_schema"]["strict"] is False
    assert captured["extra_body"] == {"future_option": {"enabled": True}}
    assert captured["extra_headers"] == {"x-openai-feature": "enabled"}
    assert fake_client.closed is True
