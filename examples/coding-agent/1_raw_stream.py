import ai
import asyncio

import inspect
import pydantic
import json

from typing import get_type_hints


def get_schema(fn) -> dict:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

    fields = {}
    for name, p in sig.parameters.items():
        t = hints.get(name, str)
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[name] = (t, default)


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-opus-4.7")

    messages = [
        ai.system_message("you are a coding assistant"),
        ai.user_message("actually i don't need assistance thanks"),
    ]

    async for e in ai.stream(model, messages):
        print(e)


if __name__ == "__main__":
    asyncio.run(main())
