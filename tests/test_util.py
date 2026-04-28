"""Tests for ai.util.merge."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import Any

import async_solipsism  # type: ignore[import-untyped]
import pytest

from ai import util


@pytest.fixture
def event_loop_policy() -> async_solipsism.EventLoopPolicy:
    return async_solipsism.EventLoopPolicy()


async def _from_list(items: list[Any], delay: float = 0) -> AsyncIterable[Any]:
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        print(asyncio.get_event_loop().time(), item)
        yield item


async def _collect(aiter: AsyncIterable[Any]) -> list[Any]:
    return [item async for item in aiter]


# -- basic behavior --------------------------------------------------------


async def test_single_iterable() -> None:
    result = await _collect(util.merge(_from_list([1, 2, 3])))
    assert result == [1, 2, 3]


async def test_empty_iterable() -> None:
    result = await _collect(util.merge(_from_list([])))
    assert result == []


async def test_no_iterables() -> None:
    result = await _collect(util.merge())
    assert result == []


async def test_multiple_iterables_all_items_yielded() -> None:
    result = await _collect(
        util.merge(
            _from_list(["a", "b"]),
            _from_list(["x", "y"]),
        )
    )
    assert sorted(result) == ["a", "b", "x", "y"]


async def test_different_lengths() -> None:
    result = await _collect(
        util.merge(
            _from_list([1, 2, 3]),
            _from_list([10]),
        )
    )
    assert sorted(result) == [1, 2, 3, 10]


async def test_many_iterables() -> None:
    result = await _collect(
        util.merge(
            _from_list([1]),
            _from_list([2]),
            _from_list([3]),
            _from_list([4]),
        )
    )
    assert sorted(result) == [1, 2, 3, 4]


# -- timing (async-solipsism) ---------------------------------------------


async def test_simulated_clock_advances() -> None:
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await _collect(
        util.merge(
            _from_list([1, 2], delay=10),
            _from_list([3, 4], delay=5),
        )
    )
    elapsed = loop.time() - t0
    assert elapsed == 20.0


async def test_ordering_shorter_delay_first() -> None:
    result = await _collect(
        util.merge(
            _from_list(["slow"], delay=100),
            _from_list(["fast"], delay=1),
        )
    )
    assert result == ["fast", "slow"]


# -- error handling --------------------------------------------------------


async def test_error_cancels_other_iterables() -> None:
    """When one iterable raises, the others are closed."""
    closed: list[str] = []

    async def good() -> AsyncIterable[str]:
        try:
            await asyncio.sleep(10)
            yield "never"
        finally:
            closed.append("good")

    async def bad() -> AsyncIterable[str]:
        await asyncio.sleep(1)
        raise RuntimeError("boom")
        yield "unreachable"  # noqa: B027

    with pytest.raises(RuntimeError, match="boom"):
        await _collect(util.merge(good(), bad()))

    assert "good" in closed


async def test_error_propagates() -> None:
    """The original exception is re-raised after cleanup."""

    async def failing() -> AsyncIterable[int]:
        yield 1
        raise ValueError("oops")

    with pytest.raises(ValueError, match="oops"):
        await _collect(util.merge(failing()))


async def test_items_before_error_are_yielded() -> None:
    """Items yielded before the error are still collected."""

    async def ok() -> AsyncIterable[str]:
        yield "a"
        await asyncio.sleep(100)
        yield "b"

    async def fails_later() -> AsyncIterable[str]:
        yield "x"
        await asyncio.sleep(1)
        raise RuntimeError("fail")

    results: list[str] = []
    with pytest.raises(RuntimeError, match="fail"):
        async for item in util.merge(ok(), fails_later()):
            results.append(item)

    assert "x" in results


async def test_cleanup_with_non_generator_iterable() -> None:
    """Iterables without aclose are handled gracefully."""

    class SimpleIter:
        def __init__(self) -> None:
            self.values = iter([1, 2])

        def __aiter__(self) -> SimpleIter:
            return self

        async def __anext__(self) -> int:
            try:
                return next(self.values)
            except StopIteration:
                raise StopAsyncIteration from None

    async def failing() -> AsyncIterable[int]:
        raise RuntimeError("boom")
        yield 0  # noqa: B027

    with pytest.raises(RuntimeError, match="boom"):
        await _collect(util.merge(SimpleIter(), failing()))
