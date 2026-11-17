import { useState, type FormEvent } from "react";

import { registerDevice, removeDevice, updateDevice } from "../api/client";
import type { Device } from "../types";

interface DevicesPageProps {
  token: string;
  devices: Device[];
  onRefresh: () => void;
}

export default function DevicesPage({ token, devices, onRefresh }: DevicesPageProps) {
  const [nodeId, setNodeId] = useState("");
  const [hostname, setHostname] = useState("");
  const [ip, setIp] = useState("");
  const [port, setPort] = useState("5001");
  const [trainingPort, setTrainingPort] = useState("5004");
  const [apiToken, setApiToken] = useState("");
  const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
  const [editHostname, setEditHostname] = useState("");
  const [editIp, setEditIp] = useState("");
  const [editPort, setEditPort] = useState("");
  const [editTrainingPort, setEditTrainingPort] = useState("");
  const [editApiToken, setEditApiToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [formMessage, setFormMessage] = useState("");
  const [formError, setFormError] = useState("");
  const [registryMessage, setRegistryMessage] = useState("");
  const [registryError, setRegistryError] = useState("");
  const managedDeviceCount = devices.filter((device) => device.is_managed).length;

  async function addDevice(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setFormError("");
    setFormMessage("");
    try {
      const device = await registerDevice(token, {
        node_id: nodeId.trim().toLowerCase(),
        hostname: hostname.trim(),
        ip: ip.trim(),
        port: Number(port),
        training_port: Number(trainingPort),
        api_token: apiToken,
      });
      const reachability = device.online
        ? "status API reachable"
        : `status API not reachable${device.last_error ? `: ${device.last_error}` : ""}`;
      const training = device.training_ready
        ? "training peers synchronized"
        : `training not ready${device.training_error ? `: ${device.training_error}` : ""}`;
      setFormMessage(`${device.hostname} registered; ${reachability}; ${training}.`);
      setNodeId("");
      setHostname("");
      setIp("");
      setPort("5001");
      setTrainingPort("5004");
      setApiToken("");
      onRefresh();
    } catch (cause) {
      setFormError(cause instanceof Error ? cause.message : "Could not register the device.");
    } finally {
      setBusy(false);
    }
  }

  async function deleteDevice(device: Device) {
    if (!window.confirm(`Remove ${device.hostname} (${device.node_id}) from the DTN?`)) return;
    setBusy(true);
    setRegistryError("");
    setRegistryMessage("");
    try {
      await removeDevice(token, device.node_id);
      setRegistryMessage(`${device.hostname} removed from the device registry.`);
      onRefresh();
    } catch (cause) {
      setRegistryError(cause instanceof Error ? cause.message : "Could not remove the device.");
    } finally {
      setBusy(false);
    }
  }

  function beginEdit(device: Device) {
    setEditingNodeId(device.node_id);
    setEditHostname(device.hostname);
    setEditIp(device.ip);
    setEditPort(String(device.port));
    setEditTrainingPort(String(device.training_port ?? "5004"));
    setEditApiToken("");
    setRegistryError("");
    setRegistryMessage("");
  }

  async function saveDevice(event: FormEvent<HTMLFormElement>, device: Device) {
    event.preventDefault();
    setBusy(true);
    setRegistryError("");
    setRegistryMessage("");
    try {
      const updates = {
        hostname: editHostname.trim(),
        ip: editIp.trim(),
        port: Number(editPort),
        training_port: Number(editTrainingPort),
        ...(editApiToken ? { api_token: editApiToken } : {}),
      };
      const updated = await updateDevice(token, device.node_id, updates);
      setEditingNodeId(null);
      setEditApiToken("");
      setRegistryMessage(
        updated.online
          ? `${updated.hostname} updated and reachable.`
          : `${updated.hostname} updated but is not reachable yet.${updated.last_error ? ` ${updated.last_error}` : ""}`,
      );
      onRefresh();
    } catch (cause) {
      setRegistryError(cause instanceof Error ? cause.message : "Could not update the device.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="content devices-page">
      <div className="page-heading">
        <h1>Manage devices</h1>
      </div>

      <div className="device-management-grid">
        <form className="device-form panel-card" onSubmit={(event) => void addDevice(event)}>
          <div className="section-heading">
            <h2>Add a worker device</h2>
          </div>
          <label htmlFor="device-id">Sequential worker ID</label>
          <input
            id="device-id"
            autoComplete="off"
            maxLength={32}
            pattern="[A-Za-z0-9][A-Za-z0-9._-]*"
            placeholder={`worker${devices.filter((device) => device.role === "worker").length + 1}`}
            required
            value={nodeId}
            onChange={(event) => setNodeId(event.target.value)}
          />
          <label htmlFor="device-name">Display name</label>
          <input
            id="device-name"
            autoComplete="off"
            maxLength={128}
            placeholder="Lab worker 1"
            required
            value={hostname}
            onChange={(event) => setHostname(event.target.value)}
          />
          <div className="device-address-fields">
            <div>
              <label htmlFor="device-ip">Private IP address</label>
              <input
                id="device-ip"
                autoComplete="off"
                inputMode="decimal"
                placeholder="127.0.0.1"
                required
                value={ip}
                onChange={(event) => setIp(event.target.value)}
              />
            </div>
            <div>
              <label htmlFor="device-port">Status port</label>
              <input
                id="device-port"
                min={1}
                max={65535}
                placeholder="5001"
                required
                type="number"
                value={port}
                onChange={(event) => setPort(event.target.value)}
              />
            </div>
            <div>
              <label htmlFor="device-training-port">Training API port</label>
              <input
                id="device-training-port"
                min={1}
                max={65535}
                required
                type="number"
                value={trainingPort}
                onChange={(event) => setTrainingPort(event.target.value)}
              />
            </div>
          </div>
          <label htmlFor="device-api-token">Device API token</label>
          <input
            id="device-api-token"
            autoComplete="new-password"
            minLength={32}
            placeholder="Device token"
            required
            type="password"
            value={apiToken}
            onChange={(event) => setApiToken(event.target.value)}
          />
          <small className="device-form-note">
            Sequential worker ID. Token is stored locally and never returned. Only private, loopback, or link-local IPs are accepted.
          </small>
          {formError && <div className="assistant-error" role="alert">{formError}</div>}
          {formMessage && <div className="device-feedback" role="status">{formMessage}</div>}
          <button className="send-button" disabled={busy} type="submit">
            {busy ? "Registering…" : "Register and test connection"}
          </button>
        </form>

        <section className="device-registry panel-card" aria-label="Registered devices">
          <div className="section-heading">
            <div>
              <h2>Registered devices</h2>
              <p>{devices.length} device{devices.length === 1 ? "" : "s"} · {managedDeviceCount} dashboard-managed</p>
            </div>
          </div>
          {devices.length === 0 ? (
            <p className="device-registry-empty">No devices are registered with the coordinator.</p>
          ) : (
            <div className="device-registry-list">
              {devices.map((device) => (
                <article className="device-registry-item" key={device.node_id}>
                  <div className="device-registry-identity">
                    <span className={`health-indicator ${device.online ? "healthy" : ""}`} />
                    <div>
                      <strong>{device.hostname}</strong>
                      <span>{device.node_id} · {device.ip}:{device.port}</span>
                    </div>
                    <span className="device-registry-kind">
                      {device.is_managed ? "Added device" : "Cluster-configured"}
                    </span>
                    <span className={`device-reachability ${device.online ? "online" : "offline"}`}>
                      {device.online ? "Reachable" : "Offline"}
                    </span>
                    {device.is_managed && (
                      <span className={`device-reachability ${device.training_ready ? "online" : "offline"}`}>
                        {device.training_ready ? "Training ready" : "Training not ready"}
                      </span>
                    )}
                  </div>
                  {device.last_error && <small className="device-last-error">{device.last_error}</small>}
                  {device.training_error && <small className="device-last-error">{device.training_error}</small>}
                  {device.is_managed ? (
                    editingNodeId === device.node_id ? (
                      <form className="device-edit-form" onSubmit={(event) => void saveDevice(event, device)}>
                        <label htmlFor={`edit-device-name-${device.node_id}`}>New display name</label>
                        <input
                          id={`edit-device-name-${device.node_id}`}
                          maxLength={128}
                          required
                          value={editHostname}
                          onChange={(event) => setEditHostname(event.target.value)}
                        />
                        <div className="device-address-fields">
                          <div>
                            <label htmlFor={`edit-device-ip-${device.node_id}`}>New private IP address</label>
                            <input
                              id={`edit-device-ip-${device.node_id}`}
                              inputMode="decimal"
                              required
                              value={editIp}
                              onChange={(event) => setEditIp(event.target.value)}
                            />
                          </div>
                          <div>
                            <label htmlFor={`edit-device-port-${device.node_id}`}>New status port</label>
                            <input
                              id={`edit-device-port-${device.node_id}`}
                              min={1}
                              max={65535}
                              required
                              type="number"
                              value={editPort}
                              onChange={(event) => setEditPort(event.target.value)}
                            />
                          </div>
                          <div>
                            <label htmlFor={`edit-device-training-port-${device.node_id}`}>New training API port</label>
                            <input
                              id={`edit-device-training-port-${device.node_id}`}
                              min={1}
                              max={65535}
                              required
                              type="number"
                              value={editTrainingPort}
                              onChange={(event) => setEditTrainingPort(event.target.value)}
                            />
                          </div>
                        </div>
                        <label htmlFor={`edit-device-token-${device.node_id}`}>New device API token (optional)</label>
                        <input
                          id={`edit-device-token-${device.node_id}`}
                          autoComplete="new-password"
                          minLength={32}
                          placeholder="Blank keeps current token"
                          type="password"
                          value={editApiToken}
                          onChange={(event) => setEditApiToken(event.target.value)}
                        />
                        <div className="device-registry-actions">
                          <button className="send-button" disabled={busy} type="submit">
                            {busy ? "Saving…" : "Save changes"}
                          </button>
                          <button
                            className="quiet-button"
                            disabled={busy}
                            onClick={() => setEditingNodeId(null)}
                            type="button"
                          >
                            Cancel
                          </button>
                        </div>
                      </form>
                    ) : (
                      <div className="device-registry-actions">
                        <button
                          className="quiet-button"
                          disabled={busy}
                          onClick={() => beginEdit(device)}
                          type="button"
                        >
                          Edit
                        </button>
                        <button
                          className="remove-device-button"
                          disabled={busy}
                          onClick={() => void deleteDevice(device)}
                          type="button"
                        >
                          Remove
                        </button>
                      </div>
                    )
                  ) : (
                    <small className="device-form-note">
                      Deployment-managed · cannot edit or remove here.
                    </small>
                  )}
                </article>
              ))}
            </div>
          )}
          {registryError && <div className="assistant-error" role="alert">{registryError}</div>}
          {registryMessage && <div className="device-feedback" role="status">{registryMessage}</div>}
        </section>
      </div>
    </section>
  );
}
