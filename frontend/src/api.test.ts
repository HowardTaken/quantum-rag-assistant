import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchHealth, fetchPapers, postQuery } from "./api";

function mockFetchOnce(body: unknown, init: { ok?: boolean; status?: number } = {}) {
  const { ok = true, status = 200 } = init;
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
  }) as unknown as typeof fetch;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("fetchHealth", () => {
  it("returns parsed health data on success", async () => {
    mockFetchOnce({ status: "ok", vectors: 180 });
    await expect(fetchHealth()).resolves.toEqual({ status: "ok", vectors: 180 });
  });

  it("throws with the backend's detail message on failure", async () => {
    mockFetchOnce({ detail: "Agent not initialized" }, { ok: false, status: 503 });
    await expect(fetchHealth()).rejects.toThrow("Agent not initialized");
  });

  it("falls back to a generic message when the error body isn't JSON", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => {
        throw new Error("not json");
      },
    }) as unknown as typeof fetch;
    await expect(fetchHealth()).rejects.toThrow("Request failed with status 500");
  });
});

describe("fetchPapers", () => {
  it("returns the papers map", async () => {
    mockFetchOnce({ papers: { "a.pdf": 3 } });
    await expect(fetchPapers()).resolves.toEqual({ papers: { "a.pdf": 3 } });
  });
});

describe("postQuery", () => {
  it("posts the question and thread id, and returns the parsed response", async () => {
    mockFetchOnce({ answer: "hi", sources: [], attempts: 0 });
    const result = await postQuery("What is X?", "thread-1");

    expect(result).toEqual({ answer: "hi", sources: [], attempts: 0 });
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/query"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: "What is X?", thread_id: "thread-1" }),
      })
    );
  });

  it("throws the backend's detail message when the query fails", async () => {
    mockFetchOnce({ detail: "Failed to process query" }, { ok: false, status: 500 });
    await expect(postQuery("bad", "t1")).rejects.toThrow("Failed to process query");
  });
});
