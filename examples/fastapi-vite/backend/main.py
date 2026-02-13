"""FastAPI application entry point."""

import fastapi
import fastapi.middleware.cors

from .routes import chat

app = fastapi.FastAPI(
    title="py-ai-fastapi-chat",
    description="Chat demo using Python Vercel AI SDK",
)

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
