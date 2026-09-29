import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ChatInput from "./ChatInput";

describe("ChatInput", () => {
  it("calls onSend with the trimmed question and clears the input", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} disabled={false} />);

    const input = screen.getByPlaceholderText(/ask about/i);
    await user.type(input, "  What is a QCL?  ");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(onSend).toHaveBeenCalledWith("What is a QCL?");
    expect(input).toHaveValue("");
  });

  it("does not call onSend for a blank or whitespace-only question", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} disabled={false} />);

    await user.type(screen.getByPlaceholderText(/ask about/i), "   ");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(onSend).not.toHaveBeenCalled();
  });

  it("disables the input and button when disabled", () => {
    render(<ChatInput onSend={vi.fn()} disabled={true} />);
    expect(screen.getByPlaceholderText(/ask about/i)).toBeDisabled();
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("disables the send button while the input is empty", () => {
    render(<ChatInput onSend={vi.fn()} disabled={false} />);
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("enables the send button once text is entered", async () => {
    const user = userEvent.setup();
    render(<ChatInput onSend={vi.fn()} disabled={false} />);

    await user.type(screen.getByPlaceholderText(/ask about/i), "hi");
    expect(screen.getByRole("button", { name: /send/i })).toBeEnabled();
  });
});
