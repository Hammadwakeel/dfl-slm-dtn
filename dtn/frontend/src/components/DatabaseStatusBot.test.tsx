import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import DatabaseStatusBot from "./DatabaseStatusBot";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("DatabaseStatusBot", () => {
  it("sends questions only to the coordinator database endpoint and displays its sources", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        answer: "worker1 is recorded as online.",
        sources: [{
          record_type: "device",
          record_id: "worker1",
          recorded_at: "2026-11-09T10:00:00+00:00",
        }],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<DatabaseStatusBot token="test-token" />);

    fireEvent.click(screen.getByRole("button", { name: "Open database status bot" }));
    fireEvent.change(screen.getByLabelText("Ask the database status bot"), {
      target: { value: "Is worker1 online?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("worker1 is recorded as online.")).toBeInTheDocument();
    expect(screen.getByText(/device · worker1/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8000/api/database-assistant/ask");
    expect(fetchMock.mock.calls[0][1]).toEqual(expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ question: "Is worker1 online?" }),
    }));
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes(":5003") || String(url).includes("/rag/"))).toBe(false);
  });
});
