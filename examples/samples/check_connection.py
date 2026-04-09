"""Check connection — verify credentials and model availability for all providers."""

import asyncio

import ai

MODELS = [
    ai.model("ai-gateway", "anthropic/claude-sonnet-4"),
    ai.model("anthropic", "claude-sonnet-4-20250514"),
    ai.model("openai", "gpt-5.4-mini"),
]


async def _check(model: ai.Model) -> None:
    try:
        ok = await ai.check_connection(model)
        status = "[OK]  " if ok else "[FAIL]"
    except Exception as exc:
        status = f"[ERR] {exc}"
    print(f"  {status}  {model.provider}/{model.id}")


async def main() -> None:
    print("Checking connections...\n")
    await asyncio.gather(*[_check(m) for m in MODELS])
    print()


if __name__ == "__main__":
    asyncio.run(main())
