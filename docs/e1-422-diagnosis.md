# E1: Topology Optimizer HTTP 422 Diagnosis

Date: 2026-11-02. Diagnosis only; no application code changed, no training started, and no worker/master endpoint was called.

## 1. Frontend Request

Component: `dtn/frontend/src/App.tsx`, `TrainingTopologyPanel.optimize()`.

Current display state is `bidirectional_ring` / `stratified`, but the optimizer request does not submit either value. It asks for the five-topology sweep with Dirichlet alphas:

```http
POST {apiBase}/api/slm/command
Content-Type: application/json
Authorization: Bearer <DTN_API_TOKEN>
```

`apiBase` is `VITE_API_BASE_URL` when configured; otherwise it is the current browser origin's protocol/hostname with port 8000 (for example `http://localhost:8000`).

Exact body built by the component and serialized by `sendSlmCommand()`:

```json
{
  "action": "optimize_topology",
  "parameters": {
    "rounds": 10,
    "epochs": 5,
    "seeds": [17, 42, 2026],
    "alphas": [0.3, 0.5]
  }
}
```

`sendSlmCommand()` sends `JSON.stringify(action)` directly, without wrapping it in an outer `action` object.

## 2. Backend Contract

The matching route is `POST /api/slm/command` in `dtn/backend/app/routers/slm.py`. It has no Pydantic request model; the parameter is `body: dict[str, object]`. Its manual checks allow only top-level `action`, `reason`, and `evidence`, then require `body.action` to be a dictionary.

The shared action schema validates `optimize_topology` parameters:

- `rounds`: optional integer, 1..100
- `epochs`: optional integer, 1..50
- `seeds`: optional array, 1..3 unique integers, each 0..4294967295
- `alphas`: optional array, 1..2 unique numbers, each exactly 0.3 or 0.5
- No other parameters are allowed.

The separate `POST /api/autonomy/dispatch` route uses Pydantic `ActionRequest`: required `action: dict`; optional `reason` defaults to `"Authenticated coordinator action"` and is limited to 4000 characters; optional `evidence` defaults to an empty dict. It is not the route called by this frontend component.

## 3. In-Process TestClient Reproduction

Reproduced with FastAPI `TestClient`, dependency overrides, and a fake controller whose `dispatch()` performs no training or network calls.

Actual frontend body to `/api/slm/command`:

```text
HTTP 422
detail: "Unexpected command fields"
```

That is the verbatim `detail` for the actual request. The top-level `parameters` key is rejected before the route reaches action validation.

For clarity, sending the same flat body to the separate Pydantic `/api/autonomy/dispatch` route gives this verbatim validation detail list:

```json
[
  {
    "type": "dict_type",
    "loc": ["body", "action"],
    "msg": "Input should be a valid dictionary",
    "input": "optimize_topology"
  }
]
```

Wrapping the existing action as `{"action": <existing action>}` returned HTTP 200 and `status: "pending"` in the same test. The fake dispatcher was called once and did not start training.

## 4. Git Evidence

The worktree was dirty before this diagnosis. Read-only `git status`, `git log`, and `git diff` show:

- Commit `ae9f18f` (`Complete Phase 4 decentralized worker training`) contains the frontend `JSON.stringify(action)` caller and the then-compatible backend proxy implementation `return _forward(settings, "/command", payload=body)`.
- The current `dtn/backend/app/routers/slm.py` has an uncommitted change replacing that proxy with autonomy dispatch. The new guard rejects unknown top-level fields, and the action now must be nested under the envelope's `action` key.
- `dtn/frontend/src/api/client.ts` remains flat and was not updated to match the changed route contract.
- The backend route’s current envelope change is uncommitted; `git log` cannot attribute it to a commit.

## 5. Root Cause and Minimal Fix

Root cause: the current uncommitted backend route expects a dispatch envelope, but the frontend still sends the historical flat action object. `parameters` is therefore an unexpected top-level field, producing HTTP 422 before training can be considered.

Outdated side: the frontend caller relative to the current backend contract. The historical frontend and historical proxy route matched; the later dirty backend route changed the contract without updating `sendSlmCommand()`.

Minimal proposal: keep the current API helper accepting a flat action, but serialize it as `{"action": action}` (and optionally include `reason`/`evidence`). Update `SlmCommandResult` and callers to handle the coordinator’s actual `pending` versus `executed` response. Do not retry or auto-approve: in default `ASK_HUMAN`, the corrected call should create a pending approval; execution requires approval under existing policy.

No fix was applied, no commit was made, and no training or external endpoint was called.

## Fix applied (E2)

`sendSlmCommand()` now serializes each flat action as `{ "action": action }`, matching `/api/slm/command`. All helper callers (`App.tsx`, `ExperimentsPage.tsx`, and `NASPage.tsx` start/stop) continue to pass flat actions; none double-wraps. Pending responses now show the approval decision ID and stop subsequent sequential commands.

Frontend tests assert the exact optimizer envelope (10 rounds, 5 epochs, seeds 17/42/2026, alphas 0.3/0.5) and a `set_topology` envelope. A backend TestClient test asserts the flat request returns 422 `Unexpected command fields`, while the wrapped request returns pending and the fake dispatcher is called once. The backend route is unchanged. No commit, backend restart, training, or worker/master endpoint call was performed.