import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import * as api from "./api";

vi.mock("./api");

describe("App", () => {
  beforeEach(() => {
    vi.mocked(api.fetchHealth).mockResolvedValue({ status: "ok", vectors: 42 });
  });

  it("shows the vector count once the health check resolves", async () => {
    render(<App />);
    expect(await screen.findByText("42 vectors ready")).toBeInTheDocument();
  });

  it("shows an unreachable message when the health check fails", async () => {
    vi.mocked(api.fetchHealth).mockRejectedValue(new Error("connection refused"));
    render(<App />);
    expect(
      await screen.findByText("Backend unreachable: connection refused")
    ).toBeInTheDocument();
  });

  it("sends a question and renders the agent's answer with its sources", async () => {
    vi.mocked(api.postQuery).mockResolvedValue({
      answer: "It covers 5.23-5.74 THz.",
      sources: [{ source: "qcl.pdf", page: "3" }],
      attempts: 0,
    });
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("42 vectors ready");
    await user.type(screen.getByPlaceholderText(/ask about/i), "What is the range?");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText("It covers 5.23-5.74 THz.")).toBeInTheDocument();
    expect(screen.getByText("qcl.pdf · p.3")).toBeInTheDocument();
    expect(api.postQuery).toHaveBeenCalledWith("What is the range?", expect.any(String));
  });

  it("shows an error bubble when the query fails instead of crashing", async () => {
    vi.mocked(api.postQuery).mockRejectedValue(new Error("Failed to process query"));
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("42 vectors ready");
    await user.type(screen.getByPlaceholderText(/ask about/i), "boom");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText("Failed to process query")).toBeInTheDocument();
  });

  it("reuses the same thread id across multiple questions in a session", async () => {
    vi.mocked(api.postQuery).mockResolvedValue({ answer: "ok", sources: [], attempts: 0 });
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("42 vectors ready");
    for (const question of ["first", "second"]) {
      await user.type(screen.getByPlaceholderText(/ask about/i), question);
      await user.click(screen.getByRole("button", { name: /send/i }));
      await screen.findAllByText("ok");
    }

    const calls = vi.mocked(api.postQuery).mock.calls;
    expect(calls[0][1]).toBe(calls[1][1]);
  });
});
