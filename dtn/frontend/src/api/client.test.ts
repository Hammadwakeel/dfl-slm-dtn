import { afterEach, describe, expect, it, vi } from "vitest";

import { dispatchAutonomyAction, sendSlmCommand } from "./client";

afterEach(() => {
  vi.unstubAllGlobals();
});

function mockPendingResponse() {
  return {
    ok: true,
    json: async () => ({ status: "pending", decision: { id: "approval-1", status: "PENDING" } }),
  };
}

describe("sendSlmCommand", () => {
  it("wraps the five-topology optimizer action in the command envelope", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockPendingResponse());
    vi.stubGlobal("fetch", fetchMock);
    const action = {
      action: "optimize_topology",
      parameters: { rounds: 10, epochs: 5, seeds: [17, 42, 2026], alphas: [0.3] },
    };

    await sendSlmCommand("test-token", action);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/slm/command",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ Authorization: "Bearer test-token" }),
        body: JSON.stringify({ action }),
      }),
    );
  });

  it("wraps a simple topology action exactly once", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockPendingResponse());
    vi.stubGlobal("fetch", fetchMock);
    const action = { action: "set_topology", parameters: { topology: "mesh" } };

    await sendSlmCommand("test-token", action);

    const [, request] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(String(request.body))).toEqual({ action });
  });
});

describe("dispatchAutonomyAction", () => {
  const action = {
    action: "enable_nas" as const,
    parameters: {
      enabled: true,
      dataset_name: "cifar10",
      search_time_limit_seconds: 600,
    },
  };

  it("normalizes the nested result returned by master-backed actions", async () => {
    const data = { accepted: true, group_id: "dataset-nas-retry" };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "executed",
        result: { action, result: { available: true, data } },
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      dispatchAutonomyAction("test-token", action, "Retry CIFAR NAS", {}),
    ).resolves.toEqual({
      status: "executed",
      result: { available: true, data },
    });
  });

  it("preserves direct results and rejects malformed success responses", async () => {
    const data = { group_id: "dataset-nas-direct" };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: "executed", result: { available: true, data } }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: "executed", result: { unexpected: true } }),
      });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      dispatchAutonomyAction("test-token", action, "Start CIFAR NAS", {}),
    ).resolves.toMatchObject({ result: { available: true, data } });
    await expect(
      dispatchAutonomyAction("test-token", action, "Start CIFAR NAS", {}),
    ).rejects.toThrow("Coordinator returned an invalid action dispatch result.");
  });
});