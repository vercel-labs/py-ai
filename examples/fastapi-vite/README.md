# fastapi-chat

Chat demo using the Python Vercel AI SDK with a FastAPI backend and React frontend.
Includes **human-in-the-loop tool approval** — every tool call is gated
behind user confirmation before execution.

## Stack

- **Backend:** FastAPI + vercel-ai-sdk (Python 3.12)
- **Frontend:** Vite + React + AI SDK UI + AI Elements

## Human-in-the-Loop

The agent graph in `backend/agent.py` uses the `ToolApproval` hook to
suspend execution whenever the LLM wants to call a tool.  The flow is:

1. LLM emits a tool call
2. Backend creates a `ToolApproval` hook — this emits an
   `approval-requested` event on the SSE stream and suspends execution
3. The frontend renders Approve / Reject buttons via the
   `<Confirmation>` component (from AI Elements)
4. When the user clicks a button, `addToolApprovalResponse()` patches
   the message and sends a new request with the decision
5. The backend resumes from the checkpoint and either executes the tool
   or marks it as denied

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
