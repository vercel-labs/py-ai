"""Middleware: run-scoped registration, chain ordering, per-surface wrapping."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic
import pytest

import ai
from ai import middleware, models
from ai.models.core.helpers import streaming as streaming_
from ai.types import messages as messages_

from .conftest import MOCK_MODEL, mock_generate, mock_llm, text_msg, tool_call_msg

# ── Helpers ──────────────────────────────────────────────────────


class Confirmation(pydantic.BaseModel):
    approved: bool
    reason: str = ""


# ── wrap_model ──────────────────────────────────────────────────


async def test_wrap_model_is_called() -> None:
    """Middleware.wrap_model is invoked for every models.stream() call."""
    model_calls: list[middleware.ModelContext] = []

    class Spy(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            model_calls.append(call)
            return await next(call)

    my_agent = ai.agent()
    mock_llm([[text_msg("Hello!")]])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], middleware=[Spy()]
    ):
        pass

    assert len(model_calls) == 1
    assert model_calls[0].model.id == "mock-model"
    assert len(model_calls[0].messages) >= 1


# ── wrap_tool ───────────────────────────────────────────────────


async def test_wrap_tool_is_called() -> None:
    """Middleware.wrap_tool is invoked for every tool execution."""
    tool_calls: list[middleware.ToolContext] = []

    class Spy(ai.Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            tool_calls.append(call)
            return await next(call)

    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    my_agent = ai.agent(tools=[double])
    call1 = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 7}')]
    call2 = [text_msg("14")]
    mock_llm([call1, call2])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Double 7")], middleware=[Spy()]
    ):
        pass

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "double"
    assert tool_calls[0].tool_call_id == "tc-1"
    assert tool_calls[0].kwargs == {"x": 7}


# ── wrap_hook ───────────────────────────────────────────────────


async def test_wrap_hook_is_called() -> None:
    """Middleware.wrap_hook is invoked for every ai.hook() call."""
    hook_calls: list[middleware.HookContext] = []

    class Spy(ai.Middleware):
        async def wrap_hook(self, call: middleware.HookContext, next: Any) -> Any:
            hook_calls.append(call)
            return await next(call)

    my_agent = ai.agent()

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.Message]:
        async for msg in await ai.models.stream(context.model, context.messages):
            yield msg
        await ai.hook("test_hook", payload=Confirmation)

    mock_llm([[text_msg("OK")]])

    async for msg in my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], middleware=[Spy()]
    ):
        if any(isinstance(p, ai.HookPart) and p.status == "pending" for p in msg.parts):
            ai.resolve_hook("test_hook", {"approved": True, "reason": "ok"})

    assert len(hook_calls) == 1
    assert hook_calls[0].label == "test_hook"
    assert hook_calls[0].payload is Confirmation
    assert hook_calls[0].interrupt_loop is False


# ── Middleware ordering (onion model) ────────────────────────────


async def test_model_middleware_ordering() -> None:
    """First in list = outermost. Sees call first, result last."""
    order: list[str] = []

    class Outer(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            order.append("outer-before")
            result = await next(call)
            order.append("outer-after")
            return result

    class Inner(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            order.append("inner-before")
            result = await next(call)
            order.append("inner-after")
            return result

    my_agent = ai.agent()
    mock_llm([[text_msg("Hi")]])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], middleware=[Outer(), Inner()]
    ):
        pass

    assert order == ["outer-before", "inner-before", "inner-after", "outer-after"]


# ── Context modification ────────────────────────────────────────


async def test_model_context_can_be_modified() -> None:
    """Middleware can modify the ModelContext before passing to next."""

    class Injector(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            # Inject a system message.
            extra = ai.system_message("Extra instruction: be concise.")
            modified = dataclasses.replace(call, messages=[extra, *call.messages])
            return await next(modified)

    # Use a capturing adapter to see what messages the LLM received.
    captured_messages: list[list[messages_.Message]] = []

    class CapturingAdapter:
        def __init__(self, responses: list[list[messages_.Message]]) -> None:
            self._responses = list(responses)
            self._idx = 0

        async def stream(
            self,
            client: Any,
            model: Any,
            messages: list[messages_.Message],
            *,
            tools: Sequence[Any] | None = None,
            output_type: type[pydantic.BaseModel] | None = None,
            **kw: Any,
        ) -> AsyncGenerator[messages_.Message]:
            captured_messages.append(list(messages))
            seq = self._responses[self._idx]
            self._idx += 1
            handler = streaming_.StreamHandler()
            for msg in seq:
                for i, part in enumerate(msg.parts):
                    if isinstance(part, messages_.TextPart):
                        bid = f"text-{i}"
                        yield handler.handle_event(streaming_.TextStart(block_id=bid))
                        if part.text:
                            yield handler.handle_event(
                                streaming_.TextDelta(block_id=bid, delta=part.text)
                            )
                        yield handler.handle_event(streaming_.TextEnd(block_id=bid))
            yield handler.handle_event(streaming_.MessageDone())

    adapter = CapturingAdapter([[text_msg("Concise!")]])
    models.register_stream("mock", adapter.stream)

    my_agent = ai.agent()
    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], middleware=[Injector()]
    ):
        pass

    # The LLM should have seen 2 messages: injected system + user.
    assert len(captured_messages) == 1
    assert len(captured_messages[0]) == 2
    assert captured_messages[0][0].role == "system"


# ── Nested agents inherit middleware ─────────────────────────────


async def test_nested_agent_extends_middleware() -> None:
    """Nested agent.run(middleware=[B]) extends, not replaces, the parent stack."""
    tags: list[str] = []

    class Tagger(ai.Middleware):
        def __init__(self, tag: str) -> None:
            self.tag = tag

        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            tags.append(self.tag)
            return await next(call)

    inner = ai.agent()

    @ai.tool  # type: ignore[arg-type]
    async def run_inner(query: str) -> AsyncGenerator[ai.Message]:
        """Run sub-agent with its own middleware."""
        async for msg in inner.run(
            MOCK_MODEL,
            [ai.user_message(query)],
            middleware=[Tagger("B")],
        ):
            yield msg

    outer = ai.agent(tools=[run_inner])

    mock_llm(
        [
            # Outer turn 1: call run_inner.
            [tool_call_msg(tc_id="tc-1", name="run_inner", args='{"query": "hi"}')],
            # Inner turn 1: text reply (consumed by inner agent).
            [text_msg("inner done", id="inner-1")],
            # Outer turn 2: final.
            [text_msg("outer done", id="outer-2")],
        ]
    )

    async for _m in outer.run(
        MOCK_MODEL, [ai.user_message("go")], middleware=[Tagger("A")]
    ):
        pass

    # Outer model calls see only A. Inner model call sees A then B (composed).
    assert tags == ["A", "A", "B", "A"]


# ── wrap_agent_run ──────────────────────────────────────────────


async def test_wrap_agent_run_ordering() -> None:
    """Agent run middleware chain runs in the correct onion order."""
    order: list[str] = []

    class Outer(ai.Middleware):
        async def wrap_agent_run(
            self, call: middleware.AgentRunContext, next: Any
        ) -> AsyncGenerator[ai.Message]:
            order.append("outer-before")
            async for msg in next(call):
                yield msg
            order.append("outer-after")

    class Inner(ai.Middleware):
        async def wrap_agent_run(
            self, call: middleware.AgentRunContext, next: Any
        ) -> AsyncGenerator[ai.Message]:
            order.append("inner-before")
            async for msg in next(call):
                yield msg
            order.append("inner-after")

    my_agent = ai.agent()
    mock_llm([[text_msg("Hi")]])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], middleware=[Outer(), Inner()]
    ):
        pass

    assert order == ["outer-before", "inner-before", "inner-after", "outer-after"]


# ── wrap_generate ───────────────────────────────────────────────


async def test_wrap_generate_is_called() -> None:
    """Middleware.wrap_generate is invoked for models.generate() inside a run."""
    gen_calls: list[middleware.GenerateContext] = []

    class Spy(ai.Middleware):
        async def wrap_generate(
            self, call: middleware.GenerateContext, next: Any
        ) -> Any:
            gen_calls.append(call)
            return await next(call)

    response = messages_.Message(
        id="gen-1",
        role="assistant",
        parts=[messages_.TextPart(text="generated image url")],
    )
    mock_generate([response])

    # Call generate inside an agent loop so middleware is active.
    my_agent = ai.agent()

    @my_agent.loop
    async def gen_loop(context: ai.Context) -> AsyncGenerator[ai.Message]:
        result = await models.generate(
            context.model, context.messages, models.ImageParams()
        )
        yield result

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("paint a cat")], middleware=[Spy()]
    ):
        pass

    assert len(gen_calls) == 1
    assert gen_calls[0].model.id == "mock-model"


async def test_wrap_tool_context_fields_flow_to_result() -> None:
    """ToolContext.tool_name is used in the result message."""

    class Rewriter(ai.Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            # Rewrite the tool_name via dataclasses.replace.
            modified = dataclasses.replace(call, tool_name="rewritten-name")
            return await next(modified)

    @ai.tool
    async def echo(x: int) -> int:
        """Echo a number."""
        return x

    my_agent = ai.agent(tools=[echo])
    call1 = [tool_call_msg(tc_id="tc-1", name="echo", args='{"x": 42}')]
    call2 = [text_msg("done")]
    mock_llm([call1, call2])

    tool_result_msgs: list[ai.Message] = []
    async for m in my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], middleware=[Rewriter()]
    ):
        if m.role == "tool" and m.tool_results:
            tool_result_msgs.append(m)

    assert len(tool_result_msgs) >= 1
    # The result message should use the rewritten name, not the original.
    assert tool_result_msgs[0].tool_results[0].tool_name == "rewritten-name"


async def test_wrap_tool_rewriting_tool_call_id_breaks_history() -> None:
    """tool_call_id is a correlation key and must stay stable."""

    class Rewriter(ai.Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            modified = dataclasses.replace(call, tool_call_id="rewritten-id")
            return await next(modified)

    @ai.tool
    async def echo(x: int) -> int:
        """Echo a number."""
        return x

    my_agent = ai.agent(tools=[echo])
    call1 = [tool_call_msg(tc_id="original-id", name="echo", args='{"x": 42}')]
    call2 = [text_msg("done")]
    mock_llm([call1, call2])

    with pytest.raises(ExceptionGroup) as exc_info:
        async for _m in my_agent.run(
            MOCK_MODEL, [ai.user_message("go")], middleware=[Rewriter()]
        ):
            pass

    assert len(exc_info.value.exceptions) == 1
    assert "orphaned-tool-result" in str(exc_info.value.exceptions[0])


# ── StreamResult wrapping ───────────────────────────────────────


async def test_middleware_can_wrap_stream_result() -> None:
    """Middleware can iterate a StreamResult and transform messages."""

    class TextAppender(ai.Middleware):
        async def wrap_model(
            self,
            call: middleware.ModelContext,
            next: Any,
        ) -> ai.StreamResultLike:
            stream_result = await next(call)

            async def _transformed() -> AsyncGenerator[messages_.Message]:
                async for msg in stream_result:
                    yield msg
                # After the stream ends, yield one more snapshot with extra text.
                yield messages_.Message(
                    id="appended",
                    role="assistant",
                    parts=[messages_.TextPart(text="original + appended")],
                )

            return ai.StreamResult.from_generator(_transformed())

    my_agent = ai.agent()
    mock_llm([[text_msg("original")]])

    msgs: list[ai.Message] = []
    async for m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], middleware=[TextAppender()]
    ):
        msgs.append(m)

    # The last message should be from the appended stream.
    texts = [m.text for m in msgs if m.text]
    assert "original + appended" in texts


# ── Context snapshot isolation ──────────────────────────────────


async def test_model_context_messages_are_isolated() -> None:
    """Mutating call.messages in middleware does not affect the caller."""
    original_messages = [ai.user_message("Hello")]

    class Mutator(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            # Try to mutate the context's messages list in place.
            call.messages.append(ai.system_message("injected"))
            return await next(call)

    my_agent = ai.agent()
    mock_llm([[text_msg("Hi")]])

    async for _m in my_agent.run(MOCK_MODEL, original_messages, middleware=[Mutator()]):
        pass

    # The original list should be unmodified.
    assert len(original_messages) == 1
    assert original_messages[0].role == "user"


# ── Middleware can repair bad tool args ──────────────────────────


async def test_middleware_can_fix_bad_tool_kwargs() -> None:
    """A middleware that rewrites call.kwargs can fix malformed tool args."""

    class ArgFixer(ai.Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            # If kwargs are empty (parse failed), supply valid ones.
            if not call.kwargs:
                fixed = dataclasses.replace(call, kwargs={"x": 99})
                return await next(fixed)
            return await next(call)

    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    my_agent = ai.agent(tools=[double])
    # Send completely invalid JSON args — parse will fail, kwargs will be {}.
    call1 = [tool_call_msg(tc_id="tc-bad", name="double", args="not json")]
    call2 = [text_msg("done")]
    mock_llm([call1, call2])

    tool_result_msgs: list[ai.Message] = []
    async for m in my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], middleware=[ArgFixer()]
    ):
        if m.role == "tool" and m.tool_results:
            tool_result_msgs.append(m)

    assert len(tool_result_msgs) >= 1
    # The fixer middleware supplied x=99, so double should return 198.
    assert tool_result_msgs[0].tool_results[0].result == 198
    assert tool_result_msgs[0].tool_results[0].is_error is False


# ── Run-scoped isolation ────────────────────────────────────────


async def test_middleware_is_run_scoped() -> None:
    """Middleware from one run does not leak into another."""
    model_calls: list[str] = []

    class Tagger(ai.Middleware):
        def __init__(self, tag: str) -> None:
            self.tag = tag

        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            model_calls.append(self.tag)
            return await next(call)

    my_agent = ai.agent()

    # Run 1: with Tagger("A")
    mock_llm([[text_msg("Hi")]])
    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], middleware=[Tagger("A")]
    ):
        pass

    # Run 2: no middleware
    mock_llm([[text_msg("Hi")]])
    async for _m in my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]):
        pass

    # Run 3: with Tagger("C")
    mock_llm([[text_msg("Hi")]])
    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], middleware=[Tagger("C")]
    ):
        pass

    assert model_calls == ["A", "C"]
