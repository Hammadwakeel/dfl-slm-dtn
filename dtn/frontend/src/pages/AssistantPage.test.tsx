import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import AssistantPage from "./AssistantPage";

const masterUrl = "http://192.168.11.15:5003";
const EMPTY_CHAT = "Ask about results, cluster health, or what to try next.";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("SLM Assistant (human mode)", () => {
  it("loads saved master conversations as plain chat without action cards", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({
          ok: true,
          json: async () => [{
            id: "conversation-saved",
            title: "What should I do next?",
            updated_at: "2026-11-03T04:00:00Z",
            message_count: 2,
          }],
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          id: "conversation-saved",
          title: "What should I do next?",
          created_at: "2026-11-03T04:00:00Z",
          updated_at: "2026-11-03T04:00:00Z",
          messages: [
            {
              id: 1,
              role: "user",
              content: "What should I do next?",
              created_at: "2026-11-03T04:00:00Z",
            },
            {
              id: 2,
              role: "assistant",
              content: "Create an initial model, then verify its distribution to all five workers.",
              created_at: "2026-11-03T04:00:02Z",
            },
          ],
        }),
      });
    }));
    render(<AssistantPage token="dtn-token" />);

    expect(await screen.findByRole("button", {
      name: "Delete chat: What should I do next?",
    })).toBeInTheDocument();
    expect(await screen.findByText(
      "Create an initial model, then verify its distribution to all five workers.",
    )).toBeInTheDocument();
    expect(screen.queryByText("Validated action")).not.toBeInTheDocument();
    expect(screen.queryByText("Action JSON")).not.toBeInTheDocument();
    expect(screen.queryByText("Technical details")).not.toBeInTheDocument();
  });

  it("sends the original query directly to the master RAG endpoint", async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          conversation_id: "conversation-1",
          reply: "The preprocessing result shows five workers received the dataset. Create the initial model next.",
        }),
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AssistantPage token="dtn-token" />);
    expect(await screen.findByText("No saved chats.")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message the SLM assistant"), {
      target: { value: "The dataset was shipped to all 5 workers. What should I do next?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText(/Create the initial model next/)).toBeInTheDocument();
    const chatCall = fetchMock.mock.calls.find(([url]) => String(url) === `${masterUrl}/rag/chat`);
    expect(chatCall).toBeDefined();
    const request = chatCall?.[1] as RequestInit;
    expect(request.method).toBe("POST");
    expect(JSON.parse(String(request.body))).toEqual({
      message: "The dataset was shipped to all 5 workers. What should I do next?",
    });
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes(":8000/api/slm/"))).toBe(false);
    expect(screen.queryByText("Validated action")).not.toBeInTheDocument();
    expect(screen.queryByText("Action JSON")).not.toBeInTheDocument();
  });

  it("keeps agent controls out of the SLM Assistant", async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AssistantPage token="dtn-token" />);
    expect(await screen.findByText("No saved chats.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Agent mode" })).not.toBeInTheDocument();
    expect(screen.queryByText("Ask questions about your cluster history. Choose and start workflows from Agent Runs.")).not.toBeInTheDocument();
    expect(screen.getByText(EMPTY_CHAT)).toBeInTheDocument();
  });

  it("deletes a master conversation and clears the active chat after confirmation", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({
          ok: true,
          json: async () => [{
            id: "conversation-saved",
            title: "Saved cluster question",
            updated_at: "2026-11-03T04:00:00Z",
            message_count: 2,
          }],
        });
      }
      if (url.endsWith("/conversation-saved") && init?.method === "DELETE") {
        return Promise.resolve({ ok: true, status: 204 });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          id: "conversation-saved",
          title: "Saved cluster question",
          created_at: "2026-11-03T04:00:00Z",
          updated_at: "2026-11-03T04:00:00Z",
          messages: [{
            id: 1,
            role: "user",
            content: "Saved cluster question",
            created_at: "2026-11-03T04:00:00Z",
          }],
        }),
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AssistantPage token="dtn-token" />);

    fireEvent.click(await screen.findByRole("button", {
      name: "Delete chat: Saved cluster question",
    }));

    expect(await screen.findByText("No saved chats.")).toBeInTheDocument();
    expect(screen.getByText(EMPTY_CHAT)).toBeInTheDocument();
    const deleteCall = fetchMock.mock.calls.find(
      ([url, init]) => String(url).endsWith("/conversation-saved") && init?.method === "DELETE",
    );
    expect(deleteCall).toBeDefined();
    expect(vi.mocked(window.confirm)).toHaveBeenCalled();
  });

  it("deletes multiple selected master conversations", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    const conversations = [
      {
        id: "conversation-first",
        title: "First saved chat",
        updated_at: "2026-11-03T04:00:00Z",
        message_count: 2,
      },
      {
        id: "conversation-second",
        title: "Second saved chat",
        updated_at: "2026-11-02T04:00:00Z",
        message_count: 2,
      },
    ];
    const deleteCalls: string[] = [];
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({ ok: true, json: async () => conversations });
      }
      if (init?.method === "DELETE") {
        deleteCalls.push(url);
        return Promise.resolve({ ok: true, status: 204 });
      }
      const conversation = conversations.find((item) => url.endsWith(`/${item.id}`));
      return Promise.resolve({
        ok: true,
        json: async () => ({
          ...conversation,
          created_at: conversation?.updated_at,
          messages: [{
            id: 1,
            role: "user",
            content: conversation?.title,
            created_at: conversation?.updated_at,
          }],
        }),
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AssistantPage token="dtn-token" />);

    fireEvent.click(await screen.findByRole("checkbox", { name: "Select all chats" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete selected" }));

    await waitFor(() => expect(deleteCalls).toHaveLength(2));
    expect(screen.getByText("No saved chats.")).toBeInTheDocument();
    expect(deleteCalls.sort()).toEqual([
      `${masterUrl}/rag/conversations/conversation-first`,
      `${masterUrl}/rag/conversations/conversation-second`,
    ]);
  });

  it("links to the Agent Supervisor for autonomous operation", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    const openAgent = vi.fn();
    render(<AssistantPage token="dtn-token" onOpenAgent={openAgent} />);
    expect(await screen.findByText("No saved chats.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Agent Supervisor" }));

    expect(openAgent).toHaveBeenCalledTimes(1);
  });
});
