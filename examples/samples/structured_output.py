import asyncio
import os

import pydantic

import vercel_ai_sdk as ai


class WeatherForecast(pydantic.BaseModel):
    city: str
    temperature: float
    conditions: str
    humidity: int
    wind_speed: float


async def main() -> None:
    # OpenAI-compatible provider (works with any model via Vercel AI Gateway)
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    # # Anthropic provider (native structured output via beta API)
    # llm = ai.anthropic.AnthropicModel(
    #     model="claude-sonnet-4-20250514",
    #     api_key=os.environ.get("ANTHROPIC_API_KEY"),
    # )

    messages = ai.make_messages(
        system="You are a weather assistant. Respond with realistic weather data.",
        user="What's the weather like in San Francisco right now?",
    )

    # Streaming: watch the JSON arrive incrementally, get validated output at the end
    print("--- Streaming ---")
    async for msg in llm.stream(messages, output_type=WeatherForecast):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
        if msg.output:
            print(f"\n\nParsed: {msg.output}")

    # Non-streaming: get the validated output directly
    print("\n--- Buffer ---")
    msg = await llm.buffer(messages, output_type=WeatherForecast)
    print(f"City: {msg.output.city}")
    print(f"Temperature: {msg.output.temperature}")
    print(f"Conditions: {msg.output.conditions}")
    print(f"Humidity: {msg.output.humidity}%")
    print(f"Wind: {msg.output.wind_speed} mph")


if __name__ == "__main__":
    asyncio.run(main())
