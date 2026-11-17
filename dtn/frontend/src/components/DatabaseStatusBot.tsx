import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

import { askDatabaseAssistant } from "../api/client";
import type { DatabaseAssistantSource } from "../api/client";

interface BotMessage {
  role: "user" | "assistant";
  text: string;
  sources?: DatabaseAssistantSource[];
}

export default function DatabaseStatusBot({ token }: { token: string }) {
  const [open, setOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<BotMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = question.trim();
    if (!text || pending) return;
    setQuestion("");
    setError("");
    setMessages((current) => [...current, { role: "user", text }]);
    setPending(true);
    try {
      const response = await askDatabaseAssistant(token, text);
      setMessages((current) => [
        ...current,
        { role: "assistant", text: response.answer, sources: response.sources },
      ]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not query the coordinator database.");
    } finally {
      setPending(false);
    }
  }

  return (
    <aside className="database-bot" aria-label="Database status bot">
      {open && (
        <section className="database-bot__panel" aria-label="Database status chat">
          <header className="database-bot__header">
            <div>
              <strong>Cluster status bot</strong>
              <span>Coordinator database only</span>
            </div>
            <button aria-label="Close status bot" onClick={() => setOpen(false)} type="button">×</button>
          </header>
          <p className="database-bot__disclaimer">
            No language model calls.
          </p>
          <div className="database-bot__messages" aria-live="polite">
            {messages.map((message, index) => (
              <article className={`database-bot__message ${message.role}`} key={`${message.role}-${index}`}>
                <strong>{message.role === "user" ? "You" : "Database"}</strong>
                <p>{message.text}</p>
                {message.sources && message.sources.length > 0 && (
                  <details>
                    <summary>Database records · {message.sources.length}</summary>
                    <ul>
                      {message.sources.map((source) => (
                        <li key={`${source.record_type}-${source.record_id}`}>
                          {source.record_type} · {source.record_id} · {source.recorded_at}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </article>
            ))}
            {pending && <p className="database-bot__pending" role="status">Reading coordinator records…</p>}
            <div ref={bottom} />
          </div>
          {error && <p className="database-bot__error" role="alert">{error}</p>}
          <form className="database-bot__composer" onSubmit={(event) => void submit(event)}>
            <textarea
              aria-label="Ask the database status bot"
              maxLength={1000}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Question…"
              rows={2}
              value={question}
            />
            <button className="primary-button" disabled={!question.trim() || pending} type="submit">
              {pending ? "Checking…" : "Ask"}
            </button>
          </form>
        </section>
      )}
      <button
        aria-expanded={open}
        aria-label={open ? "Close database status bot" : "Open database status bot"}
        className="database-bot__toggle"
        onClick={() => setOpen((current) => !current)}
        type="button"
      >
        {open ? "×" : "?"}
      </button>
    </aside>
  );
}
