import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ChatMessage, { type ChatMessageData } from "./ChatMessage";

function makeMessage(overrides: Partial<ChatMessageData> = {}): ChatMessageData {
  return { id: "1", role: "assistant", content: "Hello", ...overrides };
}

describe("ChatMessage", () => {
  it("renders the role label and content", () => {
    render(<ChatMessage message={makeMessage({ role: "user", content: "Hi there" })} />);
    expect(screen.getByText("You")).toBeInTheDocument();
    expect(screen.getByText("Hi there")).toBeInTheDocument();
  });

  it("labels assistant and error roles correctly", () => {
    const { rerender } = render(<ChatMessage message={makeMessage({ role: "assistant" })} />);
    expect(screen.getByText("Agent")).toBeInTheDocument();

    rerender(<ChatMessage message={makeMessage({ role: "error", content: "boom" })} />);
    expect(screen.getByText("Error")).toBeInTheDocument();
  });

  it("renders a source chip for each cited source", () => {
    render(
      <ChatMessage
        message={makeMessage({
          sources: [
            { source: "a.pdf", page: "3" },
            { source: "b.pdf", page: "1" },
          ],
        })}
      />
    );
    expect(screen.getByText("a.pdf · p.3")).toBeInTheDocument();
    expect(screen.getByText("b.pdf · p.1")).toBeInTheDocument();
  });

  it("renders no sources section when there are none", () => {
    render(<ChatMessage message={makeMessage({ sources: [] })} />);
    expect(screen.queryByText(/·/)).not.toBeInTheDocument();
  });

  it("shows a retry note only when attempts is greater than zero", () => {
    const { rerender } = render(<ChatMessage message={makeMessage({ attempts: 0 })} />);
    expect(screen.queryByText(/re-searched/)).not.toBeInTheDocument();

    rerender(<ChatMessage message={makeMessage({ attempts: 1 })} />);
    expect(screen.getByText(/re-searched 1 time to check grounding/)).toBeInTheDocument();

    rerender(<ChatMessage message={makeMessage({ attempts: 2 })} />);
    expect(screen.getByText(/re-searched 2 times to check grounding/)).toBeInTheDocument();
  });
});
