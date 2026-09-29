# Quantum RAG Agent — frontend

A small React + TypeScript + Vite chat UI for the [FastAPI backend](../backend/). Talks to
the API over plain `fetch`, keeps a per-tab conversation `thread_id` (so multi-turn context
works, matching the backend's per-thread memory), and shows the agent's cited sources and
whether it had to re-search before answering.

## Setup

```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_URL (default http://localhost:8000), optional VITE_API_KEY
```

`VITE_API_KEY` must match the backend's `API_KEY` env var if it has one set. It is not a
secret once built — it ships in the public JS bundle either way (see `api.ts`'s comment);
it exists to block naive scraper traffic, not to protect anything sensitive.

## Run

Start the backend first (from the repo root):

```bash
uvicorn backend.main:app --reload
```

Then, in `frontend/`:

```bash
npm run dev
```

Opens on `http://localhost:5173`. The backend's CORS allowlist (`config.py`'s `cors_origins`)
includes this by default; add your own origin via the backend's `CORS_ORIGINS` env var if
you're running the frontend somewhere else.

## Build & test

```bash
npm run build     # type-checks (tsc -b) then bundles to dist/
npm test          # 24 Vitest + React Testing Library tests
npm run preview   # serve the production build locally
```

## Layout

```
src/
  api.ts                 fetch wrappers + types mirroring backend/schemas.py
  api.test.ts
  App.tsx                chat state, thread_id, backend health check
  App.test.tsx
  components/
    ChatMessage.tsx       message bubble incl. source chips
    ChatMessage.test.tsx
    ChatInput.tsx          input box
    ChatInput.test.tsx
  setupTests.ts           RTL cleanup + jsdom scrollIntoView polyfill
  styles.css
```
