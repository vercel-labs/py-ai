"""Model params — request-scoped provider options."""

import asyncio

import ai
from ai.models.ai_gateway import GatewayStreamParams
from ai.models.ai_gateway import params as gateway_params

model = ai.ai_gateway("anthropic/claude-sonnet-4")

messages = [
    ai.system_message("Be concise."),
    ai.user_message("Explain why waterfalls look white in two sentences."),
]


async def main() -> None:
    params: ai.StreamParams[GatewayStreamParams] = [
        gateway_params.GatewayParams(sort="cost"),
        gateway_params.GatewayAnthropicParams(speed="fast"),
    ]
    async with ai.stream(model, messages, params=params) as stream:
        async for event in stream:
            if isinstance(event, ai.TextDelta):
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
