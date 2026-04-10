"""Middleware: run-scoped registration, chain ordering, per-surface wrapping."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

import ai
from ai import middleware, models
from ai.models.core.helpers import streaming as streaming_
from ai.types import messages as messages_

from .conftest import MOCK_MODEL, mock_generate, mock_llm, text_msg, tool_call_msg

# ── Helpers ──────────────────────────────────────────────────────


class Confirmation(pydantic.BaseModel):
    approved: bool
    reason: str = ""


# ── No middleware: zero-overhead pass-through ────────────────────


async def test_no_middleware_agent_runs_normally() -> None:
    """With no middleware, the agent runs exactly as before."""

    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    my_agent = ai.agent(tools=[double])
    call1 = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 5}')]
    call2 = [text_msg("The answer is 10.")]
    llm = mock_llm([call1, call2])

    msgs: list[ai.Message] = []
    async for m in my_agent.run(MOCK_MODEL, [ai.user_message("Double 5")]):
        msgs.append(m)

    assert llm.call_count == 2
    tool_results = [m for m in msgs if m.role == "tool" and m.tool_results]
    assert len(tool_results) >= 1
    assert tool_results[0].tool_results[0].result == 10


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


async def test_wrap_model_can_see_tools() -> None:
    """ModelContext.tools carries the tool list."""
    seen_tools: list[str] = []

    class Spy(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            if call.tools:
                seen_tools.extend(t.name for t in call.tools)
            return await next(call)

    @ai.tool
    async def greet(name: str) -> str:
        """Greet someone."""
        return f"Hi {name}"

    my_agent = ai.agent(tools=[greet])
    mock_llm([[text_msg("Hello!")]])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], middleware=[Spy()]
    ):
        pass

    assert "greet" in seen_tools


async def test_wrap_model_multi_turn() -> None:
    """wrap_model is called once per LLM turn in a multi-turn agent run."""
    call_count = 0

    class Counter(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return await next(call)

    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    my_agent = ai.agent(tools=[double])
    call1 = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 3}')]
    call2 = [text_msg("Done")]
    mock_llm([call1, call2])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Double 3")], middleware=[Counter()]
    ):
        pass

    assert call_count == 2


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


async def test_wrap_tool_parallel_calls() -> None:
    """Each parallel tool call gets its own middleware invocation."""
    tool_names: list[str] = []

    class Spy(ai.Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            tool_names.append(call.tool_name)
            return await next(call)

    @ai.tool
    async def alpha(x: int) -> int:
        """Alpha."""
        return x

    @ai.tool
    async def beta(x: int) -> int:
        """Beta."""
        return x

    my_agent = ai.agent(tools=[alpha, beta])

    two_tools = messages_.Message(
        id="msg-1",
        role="assistant",
        parts=[
            messages_.ToolCallPart(
                tool_call_id="tc-a",
                tool_name="alpha",
                tool_args='{"x": 1}',
                state="done",
            ),
            messages_.ToolCallPart(
                tool_call_id="tc-b",
                tool_name="beta",
                tool_args='{"x": 2}',
                state="done",
            ),
        ],
    )
    mock_llm([[two_tools], [text_msg("done", id="msg-2")]])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], middleware=[Spy()]
    ):
        pass

    assert sorted(tool_names) == ["alpha", "beta"]


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


async def test_tool_middleware_ordering() -> None:
    """Tool middleware chain runs in the correct onion order."""
    order: list[str] = []

    class Outer(ai.Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            order.append("outer-before")
            result = await next(call)
            order.append("outer-after")
            return result

    class Inner(ai.Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            order.append("inner-before")
            result = await next(call)
            order.append("inner-after")
            return result

    @ai.tool
    async def noop() -> str:
        """No-op."""
        return "ok"

    my_agent = ai.agent(tools=[noop])
    mock_llm(
        [[tool_call_msg(tc_id="tc-1", name="noop", args="{}")], [text_msg("done")]]
    )

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], middleware=[Outer(), Inner()]
    ):
        pass

    assert order == ["outer-before", "inner-before", "inner-after", "outer-after"]


# ── Partial implementation (only some methods overridden) ────────


async def test_partial_middleware_model_only() -> None:
    """Middleware that only overrides wrap_model still passes through tools."""
    model_count = 0

    class ModelOnly(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            nonlocal model_count
            model_count += 1
            return await next(call)

    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    my_agent = ai.agent(tools=[double])
    call1 = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 4}')]
    call2 = [text_msg("8")]
    mock_llm([call1, call2])

    msgs: list[ai.Message] = []
    async for m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Double 4")], middleware=[ModelOnly()]
    ):
        msgs.append(m)

    assert model_count == 2
    # Tool still executed correctly despite middleware not touching it.
    tool_results = [m for m in msgs if m.role == "tool" and m.tool_results]
    assert tool_results[0].tool_results[0].result == 8


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


# ── No middleware without agent.run() ────────────────────────────


async def test_no_middleware_outside_agent_run() -> None:
    """Standalone models.stream() calls have no middleware by default."""
    called = False

    class Spy(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            nonlocal called
            called = True
            return await next(call)

    # Even though we define a Spy, there's no way to pass it to stream()
    # directly — middleware is run-scoped.
    mock_llm([[text_msg("Hi")]])

    stream_result = await models.stream(MOCK_MODEL, [ai.user_message("Hi")])
    async for _m in stream_result:
        pass

    assert not called


# ── Nested agents inherit middleware ─────────────────────────────


async def test_nested_agent_inherits_middleware() -> None:
    """Model calls from nested agents also traverse the middleware stack."""
    model_call_count = 0

    class Counter(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            nonlocal model_call_count
            model_call_count += 1
            return await next(call)

    # Inner agent: called as a tool.
    inner = ai.agent()

    @ai.tool  # type: ignore[arg-type]  # async generator tools are valid at runtime
    async def sub_agent(query: str) -> AsyncGenerator[ai.Message]:
        """Run a sub-agent."""
        async for msg in inner.run(MOCK_MODEL, [ai.user_message(query)]):
            yield msg

    outer = ai.agent(tools=[sub_agent])

    # Outer turn 1: call sub_agent tool. Inner turn 1: text response.
    # Outer turn 2: final text.
    mock_llm(
        [
            # Outer turn 1: LLM requests sub_agent tool.
            [tool_call_msg(tc_id="tc-1", name="sub_agent", args='{"query": "hello"}')],
            # Inner turn 1: sub-agent responds with text. (registered second,
            # consumed by the inner agent's stream call)
            [text_msg("inner response", id="inner-1")],
            # Outer turn 2: final response.
            [text_msg("outer done", id="outer-2")],
        ]
    )

    async for _m in outer.run(
        MOCK_MODEL, [ai.user_message("go")], middleware=[Counter()]
    ):
        pass

    # 2 outer model calls + 1 inner model call = 3
    assert model_call_count == 3


# ── wrap_agent_run ──────────────────────────────────────────────


async def test_wrap_agent_run_is_called() -> None:
    """Middleware.wrap_agent_run is invoked when agent.run() is called."""
    run_calls: list[middleware.AgentRunContext] = []

    class Spy(ai.Middleware):
        async def wrap_agent_run(
            self, call: middleware.AgentRunContext, next: Any
        ) -> AsyncGenerator[ai.Message]:
            run_calls.append(call)
            async for msg in next(call):
                yield msg

    my_agent = ai.agent()
    mock_llm([[text_msg("Hello!")]])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], middleware=[Spy()]
    ):
        pass

    assert len(run_calls) == 1
    assert run_calls[0].model.id == "mock-model"
    assert len(run_calls[0].messages) >= 1
    assert run_calls[0].label is None


async def test_wrap_agent_run_with_label() -> None:
    """AgentRunContext carries the label from agent.run()."""
    seen_labels: list[str | None] = []

    class Spy(ai.Middleware):
        async def wrap_agent_run(
            self, call: middleware.AgentRunContext, next: Any
        ) -> AsyncGenerator[ai.Message]:
            seen_labels.append(call.label)
            async for msg in next(call):
                yield msg

    my_agent = ai.agent()
    mock_llm([[text_msg("Hello!")]])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], label="test-label", middleware=[Spy()]
    ):
        pass

    assert seen_labels == ["test-label"]


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
        result = await models.generate(context.model, context.messages)
        yield result

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("paint a cat")], middleware=[Spy()]
    ):
        pass

    assert len(gen_calls) == 1
    assert gen_calls[0].model.id == "mock-model"


async def test_wrap_generate_context_modification() -> None:
    """Middleware can modify GenerateContext before passing to next."""
    captured_messages: list[list[messages_.Message]] = []

    class Injector(ai.Middleware):
        async def wrap_generate(
            self, call: middleware.GenerateContext, next: Any
        ) -> Any:
            extra = ai.system_message("Style: watercolor")
            modified = dataclasses.replace(call, messages=[extra, *call.messages])
            return await next(modified)

    class CapturingGenerate:
        async def generate(
            self,
            client: Any,
            model: Any,
            messages: list[messages_.Message],
            params: Any = None,
        ) -> messages_.Message:
            captured_messages.append(list(messages))
            return messages_.Message(
                id="gen-1",
                role="assistant",
                parts=[messages_.TextPart(text="done")],
            )

    adapter = CapturingGenerate()
    models.register_generate("mock", adapter.generate)

    # Call generate inside an agent loop so middleware is active.
    my_agent = ai.agent()

    @my_agent.loop
    async def gen_loop(context: ai.Context) -> AsyncGenerator[ai.Message]:
        result = await models.generate(context.model, context.messages)
        yield result

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("paint")], middleware=[Injector()]
    ):
        pass

    assert len(captured_messages) == 1
    assert len(captured_messages[0]) == 2
    assert captured_messages[0][0].role == "system"


# ── Tool error through middleware ───────────────────────────────


async def test_wrap_tool_sees_error_result() -> None:
    """Middleware.wrap_tool is called even when tool args fail to parse."""
    tool_calls: list[middleware.ToolContext] = []
    results: list[ai.Message] = []

    class Spy(ai.Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            tool_calls.append(call)
            result = await next(call)
            results.append(result)
            return result

    @ai.tool
    async def strict_tool(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    my_agent = ai.agent(tools=[strict_tool])
    # Send invalid args — "x" is not an int.
    call1 = [tool_call_msg(tc_id="tc-bad", name="strict_tool", args='{"x": "abc"}')]
    call2 = [text_msg("error handled")]
    mock_llm([call1, call2])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("add")], middleware=[Spy()]
    ):
        pass

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "strict_tool"
    # The result should be an error message.
    assert len(results) == 1
    assert results[0].tool_results[0].is_error is True


async def test_wrap_tool_context_fields_flow_to_result() -> None:
    """ToolContext.tool_call_id and tool_name are used in the result message."""

    class Rewriter(ai.Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            # Rewrite the tool_call_id via dataclasses.replace.
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

    tool_result_msgs: list[ai.Message] = []
    async for m in my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], middleware=[Rewriter()]
    ):
        if m.role == "tool" and m.tool_results:
            tool_result_msgs.append(m)

    assert len(tool_result_msgs) >= 1
    # The result message should use the rewritten ID, not the original.
    assert tool_result_msgs[0].tool_results[0].tool_call_id == "rewritten-id"


# ── Hook cancellation / pre-registration through middleware ─────


async def test_wrap_hook_with_interrupt_loop() -> None:
    """wrap_hook is called for interrupt_loop=True hooks."""
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
        with contextlib.suppress(asyncio.CancelledError):
            await ai.hook("interrupt_hook", payload=Confirmation, interrupt_loop=True)

    mock_llm([[text_msg("OK")]])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], middleware=[Spy()]
    ):
        pass

    assert len(hook_calls) == 1
    assert hook_calls[0].label == "interrupt_hook"
    assert hook_calls[0].interrupt_loop is True


async def test_wrap_hook_with_pre_registration() -> None:
    """wrap_hook is called when a resolution is pre-registered."""
    hook_calls: list[middleware.HookContext] = []
    hook_results: list[Any] = []

    class Spy(ai.Middleware):
        async def wrap_hook(self, call: middleware.HookContext, next: Any) -> Any:
            hook_calls.append(call)
            result = await next(call)
            hook_results.append(result)
            return result

    my_agent = ai.agent()

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.Message]:
        async for msg in await ai.models.stream(context.model, context.messages):
            yield msg
        result = await ai.hook("pre_hook", payload=Confirmation)
        # Yield a message so we can assert the resolution worked.
        yield ai.assistant_message(f"approved={result.approved}")

    mock_llm([[text_msg("OK")]])

    # Pre-register the resolution before starting the run.
    ai.resolve_hook("pre_hook", {"approved": True, "reason": "pre-registered"})

    msgs: list[ai.Message] = []
    async for m in my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], middleware=[Spy()]
    ):
        msgs.append(m)

    assert len(hook_calls) == 1
    assert hook_calls[0].label == "pre_hook"
    assert len(hook_results) == 1
    assert hook_results[0].approved is True


# ── StreamResult wrapping ───────────────────────────────────────


async def test_middleware_can_wrap_stream_result() -> None:
    """Middleware can iterate a StreamResult and transform messages."""

    class TextAppender(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            stream_result = await next(call)
            # Wrap the StreamResult's async generator to append text.
            original_gen = stream_result._gen

            async def _transformed() -> AsyncGenerator[messages_.Message]:
                async for msg in original_gen:
                    yield msg
                # After the stream ends, yield one more snapshot with extra text.
                # Build a simple final message.
                yield messages_.Message(
                    id="appended",
                    role="assistant",
                    parts=[messages_.TextPart(text="original + appended")],
                )

            stream_result._gen = _transformed()
            return stream_result

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


async def test_model_context_tools_are_isolated() -> None:
    """Mutating call.tools in middleware does not affect the agent's tool list."""

    @ai.tool
    async def real_tool(x: int) -> int:
        """A real tool."""
        return x

    my_agent = ai.agent(tools=[real_tool])
    original_tool_count = len(my_agent._tools)

    class Mutator(ai.Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            # Try to mutate the tools list in place (runtime type is list).
            tools = call.tools
            if tools is not None:
                assert isinstance(tools, list)
                tools.append(real_tool)
            return await next(call)

    mock_llm([[text_msg("Hi")]])

    async for _m in my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], middleware=[Mutator()]
    ):
        pass

    # The agent's internal tool list should be unmodified.
    assert len(my_agent._tools) == original_tool_count


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
