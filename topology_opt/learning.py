"""APG topology learning from measurements taken on the real workers.

A short mesh *probe* run makes every worker evaluate each received peer model on
its own validation split (loss L[i, j]), record how long it waited for that model
(time T[i, j]) and report its class histogram (theta). APG then chooses, for each
receiver i, the senders whose models it should aggregate. The result is a
receiver-centric "learned" topology the workers train on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from topology_opt.apg import APGParameters, mesh_adjacency, optimize_topology, uniform_mesh_weights


class TopologyLearningError(ValueError):
    """The probe run did not produce usable APG measurements."""


def _strongly_connected(adjacency: np.ndarray) -> bool:
    """Receiver-centric adjacency: edge source -> receiver when adjacency[receiver, source]."""
    n = adjacency.shape[0]
    forward = adjacency.T.astype(bool)  # forward[source, receiver]

    def reach(matrix: np.ndarray) -> set[int]:
        seen, stack = {0}, [0]
        while stack:
            node = stack.pop()
            for nxt in np.flatnonzero(matrix[node]):
                if int(nxt) not in seen:
                    seen.add(int(nxt))
                    stack.append(int(nxt))
        return seen

    return len(reach(forward)) == n and len(reach(forward.T)) == n


def measurement_matrices(
    worker_ids: list[str],
    row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build L, T and theta from one probe round's per-worker results."""
    n = len(worker_ids)
    index = {worker: position for position, worker in enumerate(worker_ids)}
    loss = np.full((n, n), np.nan)
    wait = np.full((n, n), np.nan)
    histograms = []
    for receiver in worker_ids:
        metrics = row.get(receiver)
        if not isinstance(metrics, dict):
            raise TopologyLearningError(f"The probe has no result from {receiver}")
        for sender, value in (metrics.get("peer_losses") or {}).items():
            if sender in index and isinstance(value, (int, float)) and np.isfinite(value):
                loss[index[receiver], index[sender]] = float(value)
        for sender, value in (metrics.get("peer_arrival_seconds") or {}).items():
            if sender in index and isinstance(value, (int, float)) and np.isfinite(value):
                wait[index[receiver], index[sender]] = float(value)
        histogram = metrics.get("class_histogram")
        if not isinstance(histogram, list) or not histogram:
            raise TopologyLearningError(f"{receiver} did not report its class histogram")
        histograms.append(np.asarray(histogram, dtype=np.float64))
    if len({item.size for item in histograms}) != 1:
        raise TopologyLearningError("Workers reported class histograms of different sizes")
    off_diagonal = ~np.eye(n, dtype=bool)
    if np.isnan(loss[off_diagonal]).mean() > 0.25:
        raise TopologyLearningError("Too few peer-model losses were measured for APG")
    # A missing measurement (a peer that missed the semi-sync deadline) is treated
    # as the worst observed value, so APG avoids that link.
    for matrix in (loss, wait):
        measured = matrix[off_diagonal & ~np.isnan(matrix)]
        worst = float(measured.max()) if measured.size else 0.0
        matrix[off_diagonal & np.isnan(matrix)] = worst
        np.fill_diagonal(matrix, 0.0)
    theta = np.vstack([item / max(1.0, item.sum()) for item in histograms])
    return loss, wait, theta


def learn_senders(
    worker_ids: list[str],
    probe_row: dict[str, Any],
    parameters: APGParameters = APGParameters(),
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Run APG on probe measurements; return (receiver -> senders, diagnostics)."""
    ordered = sorted(worker_ids)
    n = len(ordered)
    if n < 3:
        raise TopologyLearningError("Topology learning needs at least three workers")
    loss, wait, theta = measurement_matrices(ordered, probe_row)
    result = optimize_topology(loss, wait, theta, uniform_mesh_weights(n), mesh_adjacency(n), parameters)
    adjacency = result.adjacency.copy()
    ring_added = False
    if not _strongly_connected(adjacency):
        # IMPLEMENTATION CHOICE: decentralized averaging only reaches consensus on a
        # strongly connected graph, so add the directed ring when APG's graph is not.
        for receiver in range(n):
            adjacency[receiver, (receiver - 1) % n] = 1
        ring_added = True
    senders = {
        ordered[receiver]: [ordered[source] for source in np.flatnonzero(adjacency[receiver])]
        for receiver in range(n)
    }
    diagnostics = {
        "objective": round(float(result.objective), 6),
        "iterations": result.iterations,
        "safeguard_hits": result.safeguard_hits,
        "ring_added_for_connectivity": ring_added,
        "edges": int(adjacency.sum()),
        "mesh_edges": n * (n - 1),
        "mean_peer_loss": round(float(loss[~np.eye(n, dtype=bool)].mean()), 6),
        "mean_wait_seconds": round(float(wait[~np.eye(n, dtype=bool)].mean()), 3),
        "weights": np.round(result.weights, 4).tolist(),
    }
    return senders, diagnostics
