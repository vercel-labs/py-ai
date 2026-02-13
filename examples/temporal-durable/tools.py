"""Tool definitions — identical to the non-durable version.

These are plain @ai.tool functions.  They know nothing about Temporal.
The durable.temporal_tool() wrapper handles routing their execution
through activities.
"""

import vercel_ai_sdk as ai


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    return {"new york": 8_336_817, "los angeles": 3_979_576}.get(
        city.lower(), 1_000_000
    )
