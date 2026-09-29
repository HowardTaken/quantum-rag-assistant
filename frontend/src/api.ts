// Mirrors backend/schemas.py -- keep these in sync with the FastAPI models.

export interface Source {
  source: string;
  page: string;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  attempts: number;
}

export interface HealthResponse {
  status: string;
  vectors: number;
}

export interface PapersResponse {
  papers: Record<string, number>;
}

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

class ApiError extends Error {}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    // response wasn't JSON; fall through to the generic message below
  }
  return `Request failed with status ${res.status}`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new ApiError(await parseErrorDetail(res));
  return res.json();
}

export async function fetchPapers(): Promise<PapersResponse> {
  const res = await fetch(`${API_URL}/papers`);
  if (!res.ok) throw new ApiError(await parseErrorDetail(res));
  return res.json();
}

export async function postQuery(question: string, threadId: string): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId }),
  });
  if (!res.ok) throw new ApiError(await parseErrorDetail(res));
  return res.json();
}
