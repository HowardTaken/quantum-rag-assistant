# ⚛ Quantum RAG Agent

A tool-using LLM agent that answers questions about a library of THz quantum-cascade-laser
and photonics papers, with citations back to the source PDF and page number. Built to speed
up literature review on my own EE research area (terahertz metasurface QCLs), and as a
project to get hands-on with agentic RAG design: tool use, self-critique/re-retrieval, and
multi-turn memory, not just a static retrieve-then-generate chain.

**Stack:** LangGraph agent · Gemini (`gemini-2.5-flash` + `gemini-embedding-001`) · ChromaDB
(persistent, local) · FastAPI backend · Streamlit / terminal front ends.

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
  earlier chain-based version didn't — the UIs stored history for display only).

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
                    app.py (Streamlit)          \
                                          (future) React frontend
```

`main.py` and `app.py` are local UIs that import the agent directly, in-process — no HTTP
hop needed for a single local user. `backend/` is a separate FastAPI service exposing the
same agent over HTTP, which is what an external client (a JS frontend, another tool, a
teammate's script) would talk to instead of importing Python.

| File | Responsibility |
|---|---|
| [`config.py`](config.py) | All settings, loaded from env/`.env` via pydantic-settings; fails fast with a clear message if `GOOGLE_API_KEY` is missing, instead of a cryptic error deep inside a LangChain call. |
| [`ingest.py`](ingest.py) | Loads PDFs from `papers/`, chunks them, embeds with Gemini, stores in Chroma. Resumable (skips already-embedded chunks) and rate-limit aware. |
| [`tools.py`](tools.py) | Vector store access + the two tools bound to the agent. |
| [`agent.py`](agent.py) | The LangGraph state machine described above. |
| [`backend/`](backend/) | FastAPI service: `POST /query`, `GET /papers`, `GET /health`. |
| [`main.py`](main.py) | Rich-based terminal chat client. |
| [`app.py`](app.py) | Streamlit web chat client. |
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
immediately without ingesting anything. To add your own papers, drop PDFs into `papers/` and
run:

```bash
python ingest.py
```

## Running it

```bash
python main.py                              # terminal chat
streamlit run app.py                        # web chat UI
uvicorn backend.main:app --reload            # HTTP API on :8000
```

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

36 tests, all hermetic — LLMs and the vector store are swapped for fakes (`tests/fakes.py`),
so the suite needs no `GOOGLE_API_KEY` and makes no network calls. Coverage: tool
formatting/edge cases, the graph's routing logic in isolation, full agent runs through fake
LLMs (the ReAct tool loop, the grade/retry loop including the max-retry cutoff, multi-turn
memory, thread isolation), and the API endpoints against a no-op startup lifespan.

Retrieval quality itself is checked separately, since it needs real embeddings:

```bash
python -m eval.retrieval_eval
```

5 hand-written questions grounded in the actual paper text, checked for whether the correct
source document shows up in the top-k retrieved chunks. Currently 5/5. Not run in CI on
purpose — it costs real API quota — but worth re-running after changing chunk size, `k`, or
the embedding model.

## Deployment

Not deployed anywhere yet. Prepped but untested (no Docker available in the environment this
was built in):

- `Dockerfile` / `docker-compose.yml` — builds the FastAPI backend with the pre-built
  `chroma_db/` baked in.
- `render.yaml` — one-click Render deploy of that image; set `GOOGLE_API_KEY` as a secret in
  the Render dashboard.
- Streamlit Cloud is the simplest path for `app.py` specifically (no Docker needed, deploys
  straight from the repo).

## Known limitations / next steps

- The Docker image uses the full `requirements.txt` (includes Streamlit/rich, unneeded by
  the API) rather than a trimmed backend-only dependency list — untrimmed because I couldn't
  verify a minimal set without being able to actually build the image.
- No React/JS frontend yet — `backend/` exists specifically so one can be added without
  touching the agent or Python UIs.
- Grading is a second LLM call per answer, which adds latency/cost; a cheaper heuristic
  (e.g. checking citation presence) could gate whether the LLM grader even runs.
- CORS on the backend is wide open (`allow_origins=["*"]`) for local development; tighten
  before pointing a real deployed frontend at a real deployed backend.
