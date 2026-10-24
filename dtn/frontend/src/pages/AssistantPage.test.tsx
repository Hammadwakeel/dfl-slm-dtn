import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import AssistantPage from "./AssistantPage";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("SLM Assistant", () => {
  it("sends chat through the coordinator and renders the validated action", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        reply: "The master is online.",
        model_reply: "Master online.",
        action: { action: "get_status", parameters: {} },
        result: {
          available: true,
          data: { hostname: "dfl-master" },
        },
        metrics: {
          latency_ms: 900,
          model_latency_ms: 850,
          system_ram_total_mb: 8192,
          system_ram_used_before_mb: 2600,
          system_ram_used_after_mb: 2700,
          system_ram_used_delta_mb: 100,
          system_ram_available_after_mb: 5492,
        },
      }),
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AssistantPage token="dtn-token" />);

    fireEvent.change(screen.getByLabelText("Message the SLM assistant"), {
      target: { value: "Check the master status" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText("The master is online.")).toBeInTheDocument();
    expect(screen.getByText("Validated action")).toBeInTheDocument();
    expect(screen.getByText("Validated action").parentElement?.parentElement)
      .toHaveTextContent(/"action": "get_status"/);
    expect(screen.getByText(/850 ms model/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/slm/chat",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ Authorization: "Bearer dtn-token" }),
        body: JSON.stringify({ message: "Check the master status" }),
      }),
    );
  });

  it("clearly marks unimplemented operations rather than implying execution", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          reply: "Training is not available yet.",
          model_reply: "Start training.",
          action: { action: "start_training", parameters: { rounds: 2, epochs: 1 } },
          result: {
            available: false,
            message: "start_training is not available yet.",
          },
          metrics: {
            latency_ms: 300,
            model_latency_ms: 250,
            system_ram_total_mb: 8192,
            system_ram_used_before_mb: 2600,
            system_ram_used_after_mb: 2650,
            system_ram_used_delta_mb: 50,
            system_ram_available_after_mb: 5542,
          },
        }),
      }),
    );
    render(<AssistantPage token="dtn-token" />);
    fireEvent.click(screen.getByRole("button", { name: /set training rounds/i }));
    expect(await screen.findByText("Not available yet")).toBeInTheDocument();
    expect(screen.getByText(/start_training is not available yet/i)).toBeInTheDocument();
  });
});
