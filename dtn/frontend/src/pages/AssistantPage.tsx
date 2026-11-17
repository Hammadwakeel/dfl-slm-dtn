import { useEffect, useRef, useState } from "react";

import {
  deleteSlmConversation,
  getSlmConversation,
  getSlmConversations,
  sendSlmMessage,
  type SlmConversationMessage,
  type SlmConversationSummary,
} from "../api/client";

interface Message {
  id: number | string;
  role: "user" | "assistant";
  text: string;
}

interface AssistantPageProps {
  token: string;
  onOpenAgent?: () => void;
}

/**
 * Human mode: the operator performs actions in the UI and asks the on-device SLM for
 * advice. The SLM answers from the master's action history and live results and never
 * executes anything. Autonomous operation lives on the Agent Supervisor page.
 */
export default function AssistantPage({ token, onOpenAgent }: AssistantPageProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<SlmConversationSummary[]>([]);
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [requestError, setRequestError] = useState("");
  const [historyError, setHistoryError] = useState("");
  const [historyLoading, setHistoryLoading] = useState(true);
  const [deletingConversationId, setDeletingConversationId] = useState<string | null>(null);
  const [selectedConversationIds, setSelectedConversationIds] = useState<Set<string>>(new Set());
  const sequence = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);
  const busy = pending || historyLoading || deletingConversationId !== null;

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  useEffect(() => {
    let active = true;
    async function loadHistory() {
      setHistoryLoading(true);
      setHistoryError("");
      try {
        const history = await getSlmConversations(token);
        if (!active) return;
        setConversations(history);
        const latest = history[0];
        if (latest) {
          setSelectedConversationId(latest.id);
          const conversation = await getSlmConversation(token, latest.id);
          if (active) setMessages(restoreMessages(conversation.messages));
        } else {
          setSelectedConversationId(null);
          setMessages([]);
        }
      } catch (cause) {
        if (active) setHistoryError(errorText(cause, "Could not load chat history."));
      } finally {
        if (active) setHistoryLoading(false);
      }
    }
    void loadHistory();
    return () => {
      active = false;
    };
  }, [token]);

  async function refreshHistory() {
    try {
      setConversations(await getSlmConversations(token));
      setHistoryError("");
    } catch (cause) {
      setHistoryError(errorText(cause, "Could not refresh chat history."));
    }
  }

  async function openConversation(conversationId: string) {
    if (pending || conversationId === selectedConversationId) return;
    setSelectedConversationId(conversationId);
    setMessages([]);
    setHistoryError("");
    setHistoryLoading(true);
    try {
      const conversation = await getSlmConversation(token, conversationId);
      setMessages(restoreMessages(conversation.messages));
    } catch (cause) {
      setHistoryError(errorText(cause, "Could not load this conversation."));
    } finally {
      setHistoryLoading(false);
    }
  }

  function startNewConversation() {
    if (busy) return;
    setSelectedConversationId(null);
    setMessages([]);
    setInput("");
    setRequestError("");
  }

  async function removeConversations(targets: SlmConversationSummary[]) {
    if (busy || targets.length === 0) return;
    const label = targets.length === 1 ? `"${targets[0].title}"` : `${targets.length} selected chats`;
    if (!window.confirm(`Permanently delete ${label} and all their messages?`)) return;
    setDeletingConversationId(targets.length === 1 ? targets[0].id : "bulk");
    setHistoryError("");
    try {
      const outcomes = await Promise.allSettled(
        targets.map((conversation) => deleteSlmConversation(token, conversation.id)),
      );
      const deletedIds = targets
        .filter((_conversation, index) => outcomes[index].status === "fulfilled")
        .map((conversation) => conversation.id);
      const remaining = conversations.filter((conversation) => !deletedIds.includes(conversation.id));
      setConversations(remaining);
      setSelectedConversationIds((current) => new Set([...current].filter((id) => !deletedIds.includes(id))));
      if (selectedConversationId && deletedIds.includes(selectedConversationId)) {
        setSelectedConversationId(null);
        setMessages([]);
        setInput("");
        if (remaining[0]) await openConversation(remaining[0].id);
      }
      const failures = outcomes.filter((outcome): outcome is PromiseRejectedResult => outcome.status === "rejected");
      if (failures.length > 0) {
        setHistoryError(
          `${deletedIds.length} deleted; ${failures.length} could not be deleted. ${failures
            .map(({ reason }) => errorText(reason, ""))
            .join(" ")}`.trim(),
        );
      }
    } finally {
      setDeletingConversationId(null);
    }
  }

  async function sendMessage(text: string) {
    const content = text.trim();
    if (!content || busy) return;
    sequence.current += 1;
    setMessages((current) => [...current, { id: `pending-user-${sequence.current}`, role: "user", text: content }]);
    setInput("");
    setPending(true);
    setRequestError("");
    try {
      const response = await sendSlmMessage(token, content, selectedConversationId ?? undefined);
      setSelectedConversationId(response.conversation_id);
      sequence.current += 1;
      setMessages((current) => [
        ...current,
        { id: `pending-assistant-${sequence.current}`, role: "assistant", text: response.reply },
      ]);
    } catch (cause) {
      setRequestError(errorText(cause, "The SLM request failed."));
    } finally {
      setPending(false);
      void refreshHistory();
    }
  }

  const allSelected = conversations.length > 0 && conversations.every((item) => selectedConversationIds.has(item.id));

  return (
    <section className="content assistant-content">
      <div className="page-heading">
        <div>
          <h1>SLM Assistant</h1>
          <p>
            Human mode: you perform the actions; the on-device SLM advises from live results and
            history and never executes anything.
            {onOpenAgent && (
              <>
                {" "}For autonomous operation use the{" "}
                <button className="link-button" type="button" onClick={onOpenAgent}>Agent Supervisor</button>.
              </>
            )}
          </p>
        </div>
        <span className="assistant-model"><span className="status-dot" /> Master SLM</span>
      </div>

      <div className="assistant-panel">
        <div className="chat-workspace">
          <aside className="chat-history" aria-label="Chat history">
            <div className="chat-history-heading">
              <div>
                <span className="eyebrow">YOUR WORKSPACE</span>
                <h2>Chat history</h2>
              </div>
              <button type="button" className="new-chat-button" disabled={busy} onClick={startNewConversation}>
                <span>＋</span> New chat
              </button>
            </div>
            {historyError && <div className="assistant-error" role="alert">{historyError}</div>}
            <div className="chat-history-list">
              {conversations.length > 0 && (
                <div className="chat-history-bulk-actions">
                  <label>
                    <input
                      aria-label="Select all chats"
                      checked={allSelected}
                      disabled={busy}
                      onChange={(event) => setSelectedConversationIds(
                        event.target.checked ? new Set(conversations.map((item) => item.id)) : new Set(),
                      )}
                      type="checkbox"
                    />
                    Select all
                  </label>
                  <span>{selectedConversationIds.size} selected</span>
                  <button
                    className="chat-history-bulk-delete"
                    disabled={busy || selectedConversationIds.size === 0}
                    onClick={() => void removeConversations(
                      conversations.filter((item) => selectedConversationIds.has(item.id)),
                    )}
                    type="button"
                  >
                    {deletingConversationId === "bulk" ? "Deleting…" : "Delete selected"}
                  </button>
                </div>
              )}
              {historyLoading && conversations.length === 0 && (
                <p className="chat-history-note" role="status">Loading saved chats…</p>
              )}
              {!historyLoading && conversations.length === 0 && !historyError && (
                <p className="chat-history-note">No saved chats.</p>
              )}
              {conversations.map((conversation) => (
                <div className="chat-history-entry" key={conversation.id}>
                  <input
                    aria-label={`Select chat: ${conversation.title}`}
                    checked={selectedConversationIds.has(conversation.id)}
                    disabled={busy}
                    onChange={(event) => setSelectedConversationIds((current) => {
                      const next = new Set(current);
                      if (event.target.checked) next.add(conversation.id);
                      else next.delete(conversation.id);
                      return next;
                    })}
                    type="checkbox"
                  />
                  <button
                    type="button"
                    className={`chat-history-item${conversation.id === selectedConversationId ? " active" : ""}`}
                    disabled={busy}
                    onClick={() => void openConversation(conversation.id)}
                    aria-current={conversation.id === selectedConversationId ? "page" : undefined}
                  >
                    <strong>{conversation.title}</strong>
                    <span>{formatDate(conversation.updated_at)}</span>
                  </button>
                  <button
                    type="button"
                    className="chat-history-delete"
                    aria-label={`Delete chat: ${conversation.title}`}
                    title="Delete conversation"
                    disabled={busy}
                    onClick={() => void removeConversations([conversation])}
                  >
                    {deletingConversationId === conversation.id ? "…" : "×"}
                  </button>
                </div>
              ))}
            </div>
          </aside>
          <div className="chat-main">
            <div className="conversation" aria-live="polite">
              {historyLoading && messages.length === 0 && (
                <div className="conversation-empty" role="status">Loading conversation…</div>
              )}
              {messages.length === 0 && !historyLoading && (
                <div className="conversation-empty">
                  Ask about results, cluster health, or what to try next.
                </div>
              )}
              {messages.map((message) => (
                <article className={`chat-message ${message.role}`} key={message.id}>
                  <div className="chat-avatar">{message.role === "assistant" ? "✦" : "H"}</div>
                  <div className="chat-message-content">
                    <span className="chat-speaker">{message.role === "assistant" ? "SLM Assistant" : "You"}</span>
                    <p>{message.text}</p>
                  </div>
                </article>
              ))}
              {pending && (
                <div className="chat-message assistant" role="status">
                  <div className="chat-avatar">✦</div>
                  <div className="chat-message-content">
                    <span className="chat-speaker">SLM Assistant</span>
                    <p className="thinking"><span /><span /><span />Thinking on dfl-master…</p>
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
                disabled={historyLoading || pending}
                maxLength={4000}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void sendMessage(input);
                  }
                }}
                placeholder="Message…"
                rows={2}
                value={input}
              />
              <div className="composer-footer">
                <span>Enter to send · Shift+Enter for a new line · max 4,000 characters</span>
                <button className="send-button" disabled={!input.trim() || busy} type="submit">
                  {pending ? "Working…" : "Send"} <span>↑</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </section>
  );
}

function restoreMessages(stored: SlmConversationMessage[]): Message[] {
  return stored.map((message) => ({ id: message.id, role: message.role, text: message.content }));
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function errorText(cause: unknown, fallback: string): string {
  return cause instanceof Error && cause.message ? cause.message : fallback;
}
