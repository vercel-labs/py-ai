"""FastAPI application entry point."""

import fastapi
import fastapi.middleware.cors

from routes import chat

api = fastapi.FastAPI(
    title="py-ai-fastapi-chat",
    description="Chat demo using Python Vercel AI SDK",
)

api.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api.include_router(chat.router)


@api.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


app = fastapi.FastAPI()
app.mount("/api", api)
