import { useEffect, useRef, useState } from "react";
import { fetchHealth, postQuery } from "./api";
import ChatInput from "./components/ChatInput";
import ChatMessage, { type ChatMessageData } from "./components/ChatMessage";
import "./styles.css";

function makeId(): string {
  return crypto.randomUUID();
}

function getThreadId(): string {
  const key = "quantum-rag-thread-id";
  try {
    const existing = sessionStorage.getItem(key);
    if (existing) return existing;
    const fresh = makeId();
    sessionStorage.setItem(key, fresh);
    return fresh;
  } catch {
    // sessionStorage can throw in private-browsing/locked-down contexts
    return makeId();
  }
}

type BackendStatus =
  | { state: "connecting" }
  | { state: "ready"; vectors: number }
  | { state: "unreachable"; message: string };

export default function App() {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<BackendStatus>({ state: "connecting" });
  const threadId = useRef(getThreadId());
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHealth()
      .then((h) => setStatus({ state: "ready", vectors: h.vectors }))
      .catch((e: unknown) =>
        setStatus({
          state: "unreachable",
          message: e instanceof Error ? e.message : "Unknown error",
        })
      );
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend(question: string) {
    setMessages((prev) => [...prev, { id: makeId(), role: "user", content: question }]);
    setLoading(true);
    try {
      const res = await postQuery(question, threadId.current);
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role: "assistant",
          content: res.answer,
          sources: res.sources,
          attempts: res.attempts,
        },
      ]);
    } catch (e: unknown) {
      const content = e instanceof Error ? e.message : "Something went wrong.";
      setMessages((prev) => [...prev, { id: makeId(), role: "error", content }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>⚛ Quantum RAG Agent</h1>
        <p className={`status status-${status.state}`}>
          {status.state === "connecting" && "Connecting to backend..."}
          {status.state === "ready" && `${status.vectors} vectors ready`}
          {status.state === "unreachable" && `Backend unreachable: ${status.message}`}
        </p>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty-state">
            Ask a question about the ingested physics papers, e.g. "What frequency range does
            the tunable metasurface QCL cover?"
          </div>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} message={m} />
        ))}
        {loading && (
          <div className="message message-assistant message-pending">
            <div className="message-role">Agent</div>
            <div className="message-content">Thinking...</div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <ChatInput onSend={handleSend} disabled={loading} />
    </div>
  );
}
