import type { Source } from "../api";

export interface ChatMessageData {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  sources?: Source[];
  attempts?: number;
}

const ROLE_LABEL: Record<ChatMessageData["role"], string> = {
  user: "You",
  assistant: "Agent",
  error: "Error",
};

export default function ChatMessage({ message }: { message: ChatMessageData }) {
  return (
    <div className={`message message-${message.role}`}>
      <div className="message-role">{ROLE_LABEL[message.role]}</div>
      <div className="message-content">{message.content}</div>
      {message.sources && message.sources.length > 0 && (
        <div className="message-sources">
          {message.sources.map((s, i) => (
            <span className="source-chip" key={`${s.source}-${s.page}-${i}`}>
              {s.source} · p.{s.page}
            </span>
          ))}
        </div>
      )}
      {!!message.attempts && (
        <div className="message-meta">
          re-searched {message.attempts} time{message.attempts > 1 ? "s" : ""} to check grounding
        </div>
      )}
    </div>
  );
}
