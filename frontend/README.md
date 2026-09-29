# Quantum RAG Agent — frontend

A small React + TypeScript + Vite chat UI for the [FastAPI backend](../backend/). Talks to
the API over plain `fetch`, keeps a per-tab conversation `thread_id` (so multi-turn context
works, matching the backend's per-thread memory), and shows the agent's cited sources and
whether it had to re-search before answering.

## Setup

```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_URL, defaults to http://localhost:8000
```

## Run

Start the backend first (from the repo root):

```bash
uvicorn backend.main:app --reload
```

Then, in `frontend/`:

```bash
npm run dev
```

Opens on `http://localhost:5173`. The backend's CORS is wide open (`allow_origins=["*"]`) for
local dev; tighten it to this app's deployed origin before shipping both publicly.

## Build

```bash
npm run build     # type-checks (tsc -b) then bundles to dist/
npm run preview   # serve the production build locally
```

## Layout

```
src/
  api.ts                 fetch wrappers + types mirroring backend/schemas.py
  App.tsx                chat state, thread_id, backend health check
  components/
    ChatMessage.tsx       message bubble incl. source chips
    ChatInput.tsx          input box
  styles.css
```
