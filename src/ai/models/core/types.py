"""Core model-layer types — parameter objects and StreamResult.

Parameter types (:class:`ImageParams`, :class:`VideoParams`) live here
because they parameterise the public :func:`generate` API.

:class:`StreamResult` is the concrete wrapper returned by :func:`stream`.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pydantic

from ...types import messages as messages_

# ---------------------------------------------------------------------------
# Generation parameter types
# ---------------------------------------------------------------------------

_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)


class ImageParams(pydantic.BaseModel):
    """Parameters for image generation (``/image-model`` endpoint)."""

    model_config = _PARAMS_CONFIG

    n: int = 1
    size: str | None = None
    aspect_ratio: str | None = pydantic.Field(
        default=None, serialization_alias="aspectRatio"
    )
    seed: int | None = None
    provider_options: dict[str, Any] = pydantic.Field(
        default_factory=dict, serialization_alias="providerOptions"
    )


class VideoParams(pydantic.BaseModel):
    """Parameters for video generation (``/video-model`` endpoint)."""

    model_config = _PARAMS_CONFIG

    n: int = 1
    aspect_ratio: str | None = pydantic.Field(
        default=None, serialization_alias="aspectRatio"
    )
    resolution: str | None = None
    duration: int | None = None
    fps: int | None = None
    seed: int | None = None
    provider_options: dict[str, Any] = pydantic.Field(
        default_factory=dict, serialization_alias="providerOptions"
    )


GenerateParams = ImageParams | VideoParams

# ---------------------------------------------------------------------------
# StreamResult
# ---------------------------------------------------------------------------


class StreamResult:
    """Wrapper around a message stream. Async-iterable; collects the final result.

    Properties like ``.text`` and ``.tool_calls`` delegate to the final
    ``Message`` snapshot and are available after iteration completes.

    When *run_id* is provided, every yielded message is stamped with it.
    When *input_messages* is provided, they are re-emitted (with *run_id*)
    before the model response stream.

    Satisfies :class:`~ai.types.StreamResultLike`.
    """

    def __init__(
        self,
        gen: AsyncGenerator[messages_.Message],
        *,
        run_id: str | None = None,
        input_messages: list[messages_.Message] | None = None,
    ) -> None:
        self._gen = gen
        self._run_id = run_id
        self._input_messages = input_messages or []
        self._final: messages_.Message | None = None

    @classmethod
    def from_generator(cls, gen: AsyncGenerator[messages_.Message]) -> StreamResult:
        """Create a :class:`StreamResult` from an async generator.

        This is the public API for middleware that needs to transform or
        replace the stream returned by ``wrap_model``::

            async def wrap_model(self, call, next):
                original = await next(call)

                async def _transformed():
                    async for msg in original:
                        yield modify(msg)

                return StreamResult.from_generator(_transformed())
        """
        return cls(gen)

    def __aiter__(self) -> AsyncGenerator[messages_.Message]:
        return self._iterate()

    async def _iterate(self) -> AsyncGenerator[messages_.Message]:
        # Re-emit input messages with run_id stamped.
        for msg in self._input_messages:
            stamped = msg.model_copy(update={"run_id": self._run_id})
            yield stamped

        # Stream model response with run_id stamped.
        async for msg in self._gen:
            if self._run_id is not None:
                msg = msg.model_copy(update={"run_id": self._run_id})
            self._final = msg
            yield msg

    @property
    def text(self) -> str:
        return self._final.text if self._final else ""

    @property
    def tool_calls(self) -> list[messages_.ToolCallPart]:
        return self._final.tool_calls if self._final else []

    @property
    def usage(self) -> messages_.Usage | None:
        return self._final.usage if self._final else None

    @property
    def output(self) -> Any:
        """Parsed structured output from the final message, if available."""
        return self._final.output if self._final else None
