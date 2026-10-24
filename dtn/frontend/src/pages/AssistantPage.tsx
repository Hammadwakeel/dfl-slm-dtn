import { useEffect, useRef, useState } from "react";

import { sendSlmMessage, type SlmChatResponse } from "../api/client";

interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  result?: SlmChatResponse;
  error?: boolean;
}

const suggestions = [
  "Check the master status",
  "Show recent assistant logs",
  "Set training rounds to 5",
];

export default function AssistantPage({ token }: { token: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [requestError, setRequestError] = useState("");
  const sequence = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  async function sendMessage(text: string) {
    const content = text.trim();
    if (!content || pending) return;
    sequence.current += 1;
    const messageId = sequence.current;
    setMessages((current) => [...current, { id: messageId, role: "user", text: content }]);
    setInput("");
    setPending(true);
    setRequestError("");
    try {
      const result = await sendSlmMessage(token, content);
      sequence.current += 1;
      setMessages((current) => [
        ...current,
        {
          id: sequence.current,
          role: "assistant",
          text: result.reply,
          result,
        },
      ]);
    } catch (cause) {
      setRequestError(cause instanceof Error ? cause.message : "The SLM request failed.");
    } finally {
      setPending(false);
    }
  }

  return (
    <section className="content assistant-content">
      <div className="page-heading">
        <div>
          <span className="eyebrow">LOCAL MODEL · LLAMA 3.2 1B</span>
          <h1>SLM Assistant</h1>
          <p>Ask the master to inspect status or propose a bounded, validated action.</p>
        </div>
        <span className="assistant-model"><span className="status-dot" /> Master connected</span>
      </div>

      <div className="assistant-panel">
        <div className="assistant-disclaimer">
          <span>i</span>
          The assistant can read live master status and service logs, and can control the supported
          Ring/Mesh CIFAR-10 experiments. Dataset upload and preprocessing are not available yet.
          It cannot execute shell commands or free-form code.
        </div>
        <div className="conversation" aria-live="polite">
          {messages.length === 0 && (
            <div className="conversation-empty">
              <div className="assistant-emblem">✦</div>
              <h2>How can I help with your cluster?</h2>
              <p>Requests are converted to one allow-listed JSON action and validated before handling.</p>
              <div className="suggestion-list">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => void sendMessage(suggestion)}
                    type="button"
                    disabled={pending}
                  >
                    {suggestion}<span>↗</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <article className={`chat-message ${message.role}`} key={message.id}>
              <div className="chat-avatar">{message.role === "assistant" ? "✦" : "H"}</div>
              <div className="chat-message-content">
                <span className="chat-speaker">{message.role === "assistant" ? "SLM Assistant" : "You"}</span>
                <p>{message.text}</p>
                {message.result && <ActionResult response={message.result} />}
              </div>
            </article>
          ))}

          {pending && (
            <div className="chat-message assistant" role="status">
              <div className="chat-avatar">✦</div>
              <div className="chat-message-content">
                <span className="chat-speaker">SLM Assistant</span>
                <p className="thinking"><span /><span /><span /> Thinking on dfl-master…</p>
              </div>
            </div>
          )}
          {requestError && <div className="assistant-error" role="alert">{requestError}</div>}
          <div ref={bottom} />
        </div>

        <form
          className="chat-composer"
          onSubmit={(event) => {
            event.preventDefault();
            void sendMessage(input);
          }}
        >
          <textarea
            aria-label="Message the SLM assistant"
            maxLength={4000}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void sendMessage(input);
              }
            }}
            placeholder="Ask about status, logs, or a training configuration…"
            rows={2}
            value={input}
          />
          <div className="composer-footer">
            <span>Enter to send · Shift+Enter for a new line · max 4,000 characters</span>
            <button className="send-button" disabled={!input.trim() || pending} type="submit">
              {pending ? "Working…" : "Send"} <span>↑</span>
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}

function ActionResult({ response }: { response: SlmChatResponse }) {
  return (
    <div className="action-result">
      <div className="action-result-heading">
        <span className="action-check">✓</span>
        <strong>Validated action</strong>
        <span className={`action-availability ${response.result.available ? "available" : "unavailable"}`}>
          {response.result.available ? "Executed" : "Not available yet"}
        </span>
      </div>
      <pre>{JSON.stringify(response.action, null, 2)}</pre>
      {!response.result.available && response.result.message && (
        <div className="action-unavailable-note">{response.result.message}</div>
      )}
      {["get_status", "get_logs"].includes(response.action.action) &&
        response.result.available && (
        <pre className="action-data">{JSON.stringify(response.result.data, null, 2)}</pre>
      )}
      <div className="inference-metrics">
        <span>{Math.round(response.metrics.model_latency_ms)} ms model</span>
        <span>{response.metrics.system_ram_used_after_mb} MB RAM used</span>
        <span>{response.metrics.system_ram_used_delta_mb >= 0 ? "+" : ""}
          {response.metrics.system_ram_used_delta_mb} MB during inference</span>
      </div>
    </div>
  );
}
