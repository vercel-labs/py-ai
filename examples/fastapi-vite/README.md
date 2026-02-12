# fastapi-chat

Chat demo using the Python Vercel AI SDK with a FastAPI backend and React frontend.

## Stack

- **Backend:** FastAPI + vercel-ai-sdk (Python 3.12)
- **Frontend:** Vite + React + AI SDK UI + AI Elements

## Setup

```bash
# Backend
cd backend
uv sync
cp .env.example .env  # add your AI_GATEWAY_API_KEY

# Frontend
cd frontend
pnpm install
```

## Development

```bash
# Terminal 1: Backend
cd backend && uv run fastapi dev main.py

# Terminal 2: Frontend
cd frontend && pnpm dev
```

The frontend dev server proxies `/api` requests to the backend at `localhost:8000`.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway API key |

## Storage

Checkpoints are persisted to `./data/` as JSON files via `FileStorage`.
The storage backend implements a simple `Storage` protocol — swap in
Redis, Postgres, or any async key-value store by providing a different
implementation.
