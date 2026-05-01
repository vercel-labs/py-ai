from __future__ import annotations

from typing import Any

import pytest

import ai
from ai import models
from ai.models.anthropic import adapter, anthropic
from ai.models.anthropic import params as anthropic_params


class _FakeUsage:
    input_tokens = 0
    output_tokens = 0

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return {"input_tokens": 0, "output_tokens": 0}


class _FakeSnapshot:
    usage = _FakeUsage()


class _EmptyAnthropicStream:
    current_message_snapshot = _FakeSnapshot()

    async def __aenter__(self) -> _EmptyAnthropicStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        return None

    def __aiter__(self) -> _EmptyAnthropicStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class _FakeMessages:
    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    def stream(self, **kwargs: Any) -> _EmptyAnthropicStream:
        self._captured.update(kwargs)
        return _EmptyAnthropicStream()


class _FakeAnthropicClient:
    def __init__(self, captured: dict[str, Any]) -> None:
        self.messages = _FakeMessages(captured)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_params_map_to_request_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_client = _FakeAnthropicClient(captured)
    monkeypatch.setattr(adapter, "_make_client", lambda client: fake_client)

    params = anthropic_params.AnthropicParams(
        thinking=anthropic_params.AnthropicEnabledThinking(
            type="enabled",
            budget_tokens=2048,
        ),
        effort="high",
        task_budget=anthropic_params.AnthropicTaskBudget(type="tokens", total=20000),
        inference_geo="us",
        metadata=anthropic_params.AnthropicMetadata(user_id="user-123"),
        cache_control=anthropic_params.AnthropicCacheControl(
            type="ephemeral", ttl="1h"
        ),
        betas=["test-beta-2026-01-01"],
        send_reasoning=False,
        disable_parallel_tool_use=True,
        extra_body={"future_option": {"enabled": True}},
        extra_headers={
            "anthropic-beta": "override-beta-2026-01-01",
            "x-anthropic-feature": "enabled",
        },
    )

    stream = adapter.stream(
        models.Client(base_url="https://anthropic.test", api_key="sk-test"),
        anthropic("claude-sonnet-4-6"),
        [
            ai.assistant_message(ai.thinking("hidden", signature="sig")),
            ai.user_message("Hi"),
        ],
        params=params,
    )
    events = [event async for event in stream]

    assert [type(event).__name__ for event in events] == ["StreamStart", "StreamEnd"]
    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert captured["output_config"] == {
        "effort": "high",
        "task_budget": {"type": "tokens", "total": 20000},
    }
    assert captured["inference_geo"] == "us"
    assert captured["metadata"] == {"user_id": "user-123"}
    assert captured["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert captured["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert captured["extra_headers"] == {
        "anthropic-beta": "override-beta-2026-01-01",
        "x-anthropic-feature": "enabled",
    }
    assert captured["extra_body"] == {"future_option": {"enabled": True}}
    assert captured["messages"] == [{"role": "user", "content": "Hi"}]
    assert fake_client.closed is True
