"""Shared fakes for the Google adapter tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from google.genai import types as genai_types


class FakeGoogleStream:
    """Async-iterable stand-in for google-genai stream chunks."""

    def __init__(self, chunks: Iterable[Any] = ()) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> FakeGoogleStream:
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeModels:
    def __init__(self, captured: dict[str, Any], stream: FakeGoogleStream) -> None:
        self._captured = captured
        self._stream = stream

    def generate_content_stream(
        self,
        *,
        model: str,
        contents: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> FakeGoogleStream:
        sdk_contents = [genai_types.Content.model_validate(item) for item in contents]
        sdk_config = (
            genai_types.GenerateContentConfig.model_validate(config)
            if config is not None
            else None
        )
        self._captured.update(
            {
                "model": model,
                "contents": contents,
                "config": config,
                "sdk_contents": sdk_contents,
                "sdk_config": sdk_config,
            }
        )
        return self._stream


class FakeGoogleClient:
    """Stand-in for ``google.genai.Client(...).aio``."""

    def __init__(
        self,
        captured: dict[str, Any] | None = None,
        stream: FakeGoogleStream | None = None,
    ) -> None:
        self.models = FakeModels(
            captured if captured is not None else {},
            stream or FakeGoogleStream(),
        )
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True
