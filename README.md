# ⚛ Quantum RAG Agent

A tool-using LLM agent that answers questions about a library of THz quantum-cascade-laser
and photonics papers, with citations back to the source PDF and page number. Built to speed
up literature review on my own EE research area (terahertz metasurface QCLs), and as a
project to get hands-on with agentic RAG design: tool use, self-critique/re-retrieval, and
multi-turn memory, not just a static retrieve-then-generate chain.

**Stack:** LangGraph agent · Gemini (`gemini-2.5-flash` + `gemini-embedding-001`) · ChromaDB
(persistent, local) · FastAPI backend · React/Vite frontend · Streamlit / terminal front ends.

## Why an agent, not a chain

The first version of this project was a standard LangChain RAG chain: embed the question,
retrieve top-k chunks, stuff them into a prompt, generate. That's fine until the question
needs a reformulated search, or the model answers confidently without actually grounding the
claim in what was retrieved (LLMs do this constantly). So the pipeline here is a small
[LangGraph](https://langchain-ai.github.io/langgraph/) state machine instead:

```
START -> agent -> [tool call?] -> tools -> agent  (ReAct loop)
                -> [no tool call] -> grade -> [ungrounded, retries left] -> agent
                                           -> [grounded / out of retries] -> END
```

- **`agent`** is a normal ReAct node: the LLM sees two tools, `search_papers` (vector search
  over the corpus) and `list_available_papers` (corpus metadata, no search needed), and
  decides whether to call one, or answer directly.
- **`grade`** runs after the agent produces a candidate answer with no further tool calls: a
  second LLM call judges whether that answer's claims are actually supported by what came
  back from `search_papers`. If not, it injects a corrective instruction and loops the agent
  back to search again with a reformulated query — bounded by `max_agent_iterations` (default
  2) so it can't loop forever.
- Conversation memory is real: each session gets a `thread_id`, and a LangGraph checkpointer
  persists message history per thread, so follow-up questions actually have context (the
  earlier chain-based version didn't — the UIs stored history for display only). The
  backend uses a SQLite-backed checkpointer (`checkpointer.py`), not the default in-memory
  one — see [Persistence, auth, and rate limiting](#persistence-auth-and-rate-limiting).

See [agent.py](agent.py) for the graph and [tools.py](tools.py) for the tool implementations.

## Architecture

```
papers/*.pdf --ingest.py--> chroma_db/ (persistent vector store)
                                  |
                        tools.py (search_papers, list_available_papers)
                                  |
                             agent.py (LangGraph agent)
                              /          \
                    main.py (terminal)   backend/ (FastAPI: /health /papers /query)
                    app.py (Streamlit)          |
                                          frontend/ (React/Vite SPA)
```

`main.py` and `app.py` are local UIs that import the agent directly, in-process — no HTTP
hop needed for a single local user. `backend/` is a separate FastAPI service exposing the
same agent over HTTP; `frontend/` is a React SPA that talks to it over `fetch`, and is what
an external client (or a teammate's script) would model itself on instead of importing
Python.

| File | Responsibility |
|---|---|
| [`config.py`](config.py) | All settings, loaded from env/`.env` via pydantic-settings; fails fast with a clear message if `GOOGLE_API_KEY` is missing, instead of a cryptic error deep inside a LangChain call. |
| [`ingest.py`](ingest.py) | Loads PDFs from `papers/`, chunks them, embeds with Gemini, stores in Chroma. Resumable (skips already-embedded chunks) and rate-limit aware. |
| [`tools.py`](tools.py) | Vector store access + the two tools bound to the agent. |
| [`agent.py`](agent.py) | The LangGraph state machine described above. |
| [`checkpointer.py`](checkpointer.py) | SQLite-backed conversation persistence. |
| [`backend/`](backend/) | FastAPI service: `POST /query`, `GET /papers`, `GET /health`. |
| [`frontend/`](frontend/) | React + TypeScript + Vite chat UI for the backend. |
| [`main.py`](main.py) | Rich-based terminal chat client. |
| [`app.py`](app.py) | Streamlit web chat client, plus a sidebar uploader for adding papers locally. |
| [`eval/retrieval_eval.py`](eval/retrieval_eval.py) | Manual retrieval-quality check (see below). |
| [`tests/`](tests/) | Hermetic pytest suite — no network, no real API key needed. |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # or `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
cp .env.example .env         # then fill in GOOGLE_API_KEY
```

A pre-built `chroma_db/` for the three included papers is committed, so you can run
immediately without ingesting anything.

**Adding more papers** — two ways:

- **CLI:** drop PDFs into `papers/` and run `python ingest.py`. Resumable (skips
  already-embedded chunks) and respects Gemini's rate limits.
- **Streamlit sidebar** (easier if you're doing this more than once): `streamlit run app.py`,
  then use the "Add papers" uploader in the sidebar. It saves the PDFs into `papers/` and
  runs the same ingestion logic, then reloads the vector store so the new papers are
  searchable immediately in that session. This is local-only by design — uploads were
  deliberately never added to the public API/frontend (see
  [Persistence, auth, and rate limiting](#persistence-auth-and-rate-limiting) for why
  accepting arbitrary uploads from the public internet is a different risk than the
  read-only `/query` endpoint), so this only affects your own machine.

Either way, `chroma_db/` (and any new PDFs) need to be committed and pushed for the changes
to reach the deployed backend — Render rebuilds from whatever's in the repo, it doesn't
ingest anything live.

## Running it

```bash
python main.py                              # terminal chat
streamlit run app.py                        # web chat UI
uvicorn backend.main:app --reload            # HTTP API on :8000
```

For the React frontend, run the backend as above, then in a separate terminal:

```bash
cd frontend
npm install
npm run dev                                  # dev server on :5173
```

See [frontend/README.md](frontend/README.md) for details. Verified end-to-end on this
machine: `npm run build` type-checks and bundles clean, and the dev server's cross-origin
requests to the backend (CORS preflight + `POST /query`) work as the browser would make them.

Example API call:

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the maximum tuning frequency of the metasurface QCL?", "thread_id": "demo"}'
```

```json
{
  "answer": "The maximum operating frequency achieved for the metasurface QCL is 5.74 THz, with single-mode tuning demonstrated from 5.23-5.73 THz. This corresponds to a tuning bandwidth of approximately 500 GHz (9.1% fractional tuning).\n\nSource: tunable-metasurface-external-cavity-quantum-cascade-lasers-up-to-5-74-thz.pdf, pages 0, 3, 5.",
  "sources": [
    {"source": "tunable-metasurface-external-cavity-quantum-cascade-lasers-up-to-5-74-thz.pdf", "page": "0"},
    {"source": "tunable-metasurface-external-cavity-quantum-cascade-lasers-up-to-5-74-thz.pdf", "page": "3"},
    {"source": "tunable-metasurface-external-cavity-quantum-cascade-lasers-up-to-5-74-thz.pdf", "page": "1"},
    {"source": "tunable-metasurface-external-cavity-quantum-cascade-lasers-up-to-5-74-thz.pdf", "page": "5"}
  ],
  "attempts": 0
}
```

## Testing

```bash
pytest
```

55 tests, all hermetic — LLMs and the vector store are swapped for fakes (`tests/fakes.py`),
so the suite needs no `GOOGLE_API_KEY` and makes no network calls. Coverage: tool
formatting/edge cases, the graph's routing logic in isolation, full agent runs through fake
LLMs (the ReAct tool loop, the grade/retry loop including the max-retry cutoff, multi-turn
memory, thread isolation), the SQLite checkpointer surviving a closed/reopened connection,
the API endpoints (auth required/skipped, rate-limit 429s, 503s before startup) against a
no-op startup lifespan, and the paper-upload helper (path-traversal-safe filename handling,
non-PDF rejection). Runs in CI on every push/PR via
[`.github/workflows/test-backend.yml`](.github/workflows/test-backend.yml).

Frontend:

```bash
cd frontend && npm test
```

24 Vitest + React Testing Library tests: the typed `api.ts` client (success/error paths,
malformed error bodies, the `X-API-Key` header being sent only when configured), both
components (role labels, source chips, disabled/enabled states, the retry note), and an
`App`-level integration suite that mocks the API module to verify the full flow (health
check, sending a question, rendering sources, error bubbles, thread-id reuse across turns)
without a real backend. Also runs in CI, via
[`.github/workflows/test-frontend.yml`](.github/workflows/test-frontend.yml).

Retrieval quality itself is checked separately, since it needs real embeddings:

```bash
python -m eval.retrieval_eval
```

5 hand-written questions grounded in the actual paper text, checked for whether the correct
source document shows up in the top-k retrieved chunks. Currently 5/5. Not run in CI on
purpose — it costs real API quota — but worth re-running after changing chunk size, `k`, or
the embedding model.

## Deployment

Both pieces are live:

- **Frontend:** [howardtaken.github.io/quantum-rag-assistant](https://howardtaken.github.io/quantum-rag-assistant/)
  — deployed automatically by [`.github/workflows/deploy-frontend.yml`](.github/workflows/deploy-frontend.yml)
  on every push to `master` that touches `frontend/`. GitHub Pages is configured with
  `build_type: workflow`, so there's no separate hosting account.
- **Backend:** `https://quantum-rag-agent-api.onrender.com` (`/health`, `/papers`, `/query`)
  — deployed via Render's dashboard (Blueprint, from `render.yaml`), which built the
  `Dockerfile` for real and confirmed it works (this had been untested locally, no Docker in
  the dev environment). Render's free tier spins down on inactivity, so the first request
  after idle can take 30-60s to wake it up.

The frontend's build bakes in the backend URL via a GitHub Actions repo variable:

```bash
gh variable set VITE_API_URL --body "https://quantum-rag-agent-api.onrender.com"
```

Re-run the `Deploy frontend to GitHub Pages` workflow (or push any change under `frontend/`)
to rebuild with a different URL. The backend's CORS (`config.py`'s `cors_origins`,
overridable via a `CORS_ORIGINS` env var) allows the GitHub Pages origin by default.

Verified end-to-end with curl, not just "should work": CORS preflight from the real GitHub
Pages origin succeeds, a real `/query` call returns a correct answer, and the deployed
frontend's JS bundle was checked to actually contain the deployed backend's URL.

Streamlit Cloud deploy for `app.py` was considered and skipped on purpose — the React
frontend + API is the real deployment target; a second hosted UI would just be a second place
for the API key to live for no real benefit.

## Persistence, auth, and rate limiting

**Conversation persistence.** The backend uses a SQLite-backed LangGraph checkpointer
(`checkpointer.py`, `settings.checkpoint_db_path`, default `checkpoints.sqlite`) instead of
the default in-memory one. That survives the process crashing and restarting without a full
redeploy — a real difference, verified with a test that closes and reopens the SQLite
connection and confirms prior messages are still there
([`tests/test_checkpointer.py`](tests/test_checkpointer.py)). It does *not* survive a Render
free-tier redeploy (ephemeral disk, wiped on every deploy) and would *not* be shared across
multiple horizontally-scaled instances — only an external store (Postgres, Redis) fixes
those; `langgraph-checkpoint-postgres` is a drop-in swap for `checkpointer.py` if this ever
needs to scale past one instance.

**Auth.** `/query` and `/papers` require an `X-API-Key` header matching the `API_KEY` env var
(`/health` stays open, for uptime monitoring). This is explicitly a *soft* gate, not real
authentication: the key ships inside the public frontend's JS bundle since there's no
server-side proxy to keep it secret behind, so anyone who opens devtools can read it. Its
job is narrower — block naive scrapers and randomly-discovered-URL traffic from spending the
backend's Gemini quota, not stop a determined attacker. If `API_KEY` isn't set (e.g. local
dev), the check is skipped entirely — see `require_api_key()` in `backend/main.py`.

**Rate limiting.** Per-IP limits via [slowapi](https://github.com/laurentS/slowapi):
`RATE_LIMIT_QUERY` (default `5/hour`) on `/query`, `RATE_LIMIT_DEFAULT` (default `60/minute`)
on everything else. The `5/hour` default isn't a round number — while building this I
exhausted Gemini's actual free-tier quota for `gemini-2.5-flash` doing manual testing, which
turned out to be **20 requests/day, total, shared across every user of the deployment**. A
per-IP rate limit generous enough to feel unrestrictive for one person is still nearly
meaningless against that shared daily cap; the real ceiling on this deployment is Google's
quota, not slowapi. This limit's actual job is narrower: stop one IP from single-handedly
burning the whole day's budget before anyone else gets a turn.

To reproduce this setup: set `API_KEY` on Render (dashboard secret) and as the
`VITE_API_KEY` GitHub Actions repo variable (`gh variable set VITE_API_KEY --body <value>` —
it's not a secret once it's in the bundle either way, see above), then redeploy both.

## Known limitations / next steps

- The Docker image uses the full `requirements.txt` (includes Streamlit/rich, unneeded by
  the API) rather than a trimmed backend-only dependency list. It does build and run
  correctly (Render built it for the live deployment), so this is a size/cold-start
  optimization, not a correctness gap.
- The frontend's tests mock the API client and were never clicked through in an actual
  browser (no browser automation tool was available) — verified via component/integration
  tests, `npm run build`'s type-check, and replaying the real cross-origin requests with
  curl against the live backend, but none of that is the same as eyeballing the rendered UI.
- Grading is a second LLM call per answer, which adds latency/cost; a cheaper heuristic
  (e.g. checking citation presence) could gate whether the LLM grader even runs.
- Auth, rate limiting, and persistent conversation state are implemented (see
  [Persistence, auth, and rate limiting](#persistence-auth-and-rate-limiting) above), but
  with real caveats worth restating: the API key is a soft gate (it's in the public JS
  bundle, not a real secret), and SQLite persistence doesn't survive a Render free-tier
  redeploy or scale across multiple instances — both would need an external store
  (Postgres/Redis) to actually fix.
