"""Accelerated projected-gradient topology selection for receiver-centric graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class APGParameters:
    alpha: float = 0.6
    beta: float = 0.4
    gamma: float = 0.6
    mu: float = 0.2
    lambda_: float = 0.2
    psi: float = 0.01
    epsilon: float = 0.15
    max_iterations: int = 100

    def __post_init__(self) -> None:
        if not np.isclose(self.alpha + self.beta, 1.0):
            raise ValueError("APG alpha and beta must sum to one")
        if not np.isclose(self.gamma + self.mu + self.lambda_, 1.0):
            raise ValueError("APG gamma, mu, and lambda must sum to one")
        if min(self.alpha, self.beta, self.gamma, self.mu, self.lambda_) < 0:
            raise ValueError("APG objective weights cannot be negative")
        if self.psi < 0 or not 0 < self.epsilon < 1:
            raise ValueError("APG psi or epsilon is outside its valid range")
        if self.max_iterations < 1:
            raise ValueError("APG max_iterations must be positive")


@dataclass(frozen=True)
class APGResult:
    weights: np.ndarray
    adjacency: np.ndarray
    objective: float
    iterations: int
    safeguard_hits: int
    objective_history: tuple[float, ...]
    lipschitz_constant: float
    step_size: float


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Project a vector onto {x >= 0, sum(x) = 1}."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("Simplex projection requires a finite non-empty vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1, dtype=np.float64)
    valid = ordered - cumulative / indices > 0
    rho = int(np.flatnonzero(valid)[-1])
    threshold = cumulative[rho] / (rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    projected /= projected.sum()
    return projected


def project_rows_to_simplex(matrix: np.ndarray) -> np.ndarray:
    """Project every off-diagonal row onto its probability simplex."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError("APG projection requires a square matrix with at least two nodes")
    if not np.isfinite(values).all():
        raise ValueError("APG projection input must be finite")
    projected = np.zeros_like(values)
    for row_index in range(values.shape[0]):
        columns = np.arange(values.shape[0]) != row_index
        projected[row_index, columns] = project_simplex(values[row_index, columns])
    return projected


def normalize_off_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Min-max normalize measured off-diagonal entries; a constant range maps to zero."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("APG inputs must be square matrices")
    normalized = np.zeros_like(values)
    mask = ~np.eye(values.shape[0], dtype=bool)
    measured = values[mask]
    if not np.isfinite(measured).all():
        raise ValueError("APG off-diagonal inputs must be finite; stale measurements must be filled first")
    lower = float(measured.min())
    upper = float(measured.max())
    if upper == lower:
        return normalized
    normalized[mask] = (measured - lower) / (upper - lower)
    return normalized


def objective_value(
    weights: np.ndarray,
    loss_normalized: np.ndarray,
    time_normalized: np.ndarray,
    theta: np.ndarray,
    previous_weights: np.ndarray,
    parameters: APGParameters,
) -> float:
    n = weights.shape[0]
    global_mix = np.tile(theta.mean(axis=0, keepdims=True), (n, 1))
    objective_linear = float(
        np.sum((parameters.alpha * loss_normalized + parameters.beta * time_normalized) * weights)
    )
    objective_balance = float(np.sum((weights @ theta - global_mix) ** 2) / n)
    objective_change = float(np.sum((weights - previous_weights) ** 2))
    return float(
        parameters.gamma * objective_linear
        + parameters.mu * objective_balance
        + parameters.lambda_ * objective_change
    )


def objective_gradient(
    weights: np.ndarray,
    loss_normalized: np.ndarray,
    time_normalized: np.ndarray,
    theta: np.ndarray,
    previous_weights: np.ndarray,
    parameters: APGParameters,
) -> np.ndarray:
    n = weights.shape[0]
    global_mix = np.tile(theta.mean(axis=0, keepdims=True), (n, 1))
    return (
        parameters.gamma * (parameters.alpha * loss_normalized + parameters.beta * time_normalized)
        + (2.0 * parameters.mu / n) * ((weights @ theta - global_mix) @ theta.T)
        + 2.0 * parameters.lambda_ * (weights - previous_weights)
    )


def gradient_lipschitz_constant(theta: np.ndarray, parameters: APGParameters) -> float:
    spectral_norm = float(np.linalg.norm(np.asarray(theta, dtype=np.float64), ord=2))
    # Section 5 default: the stated gradient Lipschitz upper bound.
    return float((2.0 * parameters.mu / theta.shape[0]) * spectral_norm**2 + 2.0 * parameters.lambda_)


def threshold_weights(weights: np.ndarray, epsilon: float) -> tuple[np.ndarray, int]:
    """Threshold W; preserve one strongest off-diagonal edge for every empty row."""
    values = np.asarray(weights, dtype=np.float64)
    adjacency = (values >= epsilon).astype(np.int8)
    np.fill_diagonal(adjacency, 0)
    safeguard_hits = 0
    for row_index in range(values.shape[0]):
        if not adjacency[row_index].any():
            candidate_values = values[row_index].copy()
            candidate_values[row_index] = -np.inf
            # IMPLEMENTATION CHOICE: argmax returns the first index, making ties deterministic.
            adjacency[row_index, int(np.argmax(candidate_values))] = 1
            safeguard_hits += 1
    return adjacency, safeguard_hits


def uniform_mesh_weights(node_count: int) -> np.ndarray:
    if node_count < 2:
        raise ValueError("APG mesh initialization requires at least two nodes")
    weights = np.full((node_count, node_count), 1.0 / (node_count - 1), dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    return weights


def mesh_adjacency(node_count: int) -> np.ndarray:
    adjacency = np.ones((node_count, node_count), dtype=np.int8)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def ring_adjacency(node_count: int) -> np.ndarray:
    """Receiver-centric directed ring: row i receives from its previous node."""
    if node_count < 3:
        raise ValueError("Ring topology requires at least three nodes")
    adjacency = np.zeros((node_count, node_count), dtype=np.int8)
    for receiver in range(node_count):
        adjacency[receiver, (receiver - 1) % node_count] = 1
    return adjacency


def map_real_worker_sender_edges(topology: str) -> set[tuple[str, str]]:
    """Map receiver-centric worker sender relationships onto node_5..node_7 IDs."""
    from worker.training_service import topology_senders

    worker_ids = ("worker1", "worker2", "worker3")
    worker_to_node = {worker: f"node_{index + 5}" for index, worker in enumerate(worker_ids)}
    return {
        (worker_to_node[receiver], worker_to_node[sender])
        for receiver in worker_ids
        for sender in topology_senders(receiver, worker_ids, topology)
    }


def adjacency_edges(adjacency: np.ndarray) -> list[tuple[int, int]]:
    """Return (receiver, selected-source) pairs in receiver-centric semantics."""
    return [
        (int(receiver), int(source))
        for receiver, source in np.argwhere(np.asarray(adjacency) != 0)
    ]


def count_edge_changes(previous: np.ndarray, current: np.ndarray) -> int:
    if previous.shape != current.shape:
        raise ValueError("Adjacency matrices must have the same shape")
    return int(np.count_nonzero(np.asarray(previous) != np.asarray(current)))


def optimize_topology(
    loss_matrix: np.ndarray,
    wait_matrix: np.ndarray,
    theta: np.ndarray,
    previous_weights: np.ndarray,
    previous_adjacency: np.ndarray,
    parameters: APGParameters = APGParameters(),
) -> APGResult:
    loss = np.asarray(loss_matrix, dtype=np.float64)
    wait = np.asarray(wait_matrix, dtype=np.float64)
    class_mix = np.asarray(theta, dtype=np.float64)
    w_prev = np.asarray(previous_weights, dtype=np.float64)
    adjacency_prev = np.asarray(previous_adjacency)
    n = w_prev.shape[0]
    if loss.shape != (n, n) or wait.shape != (n, n):
        raise ValueError("L, T, and W must have matching n x n shapes")
    if class_mix.ndim != 2 or class_mix.shape[0] != n or not np.isfinite(class_mix).all():
        raise ValueError("Theta must be a finite n x k matrix")
    if adjacency_prev.shape != (n, n):
        raise ValueError("Previous adjacency must have shape n x n")

    loss_normalized = normalize_off_diagonal(loss)
    time_normalized = normalize_off_diagonal(wait)
    lipschitz = gradient_lipschitz_constant(class_mix, parameters)
    step_size = 1.0 / lipschitz

    # Section 5 choices: W0 starts from W_prev; k0=1; M=100; eta=1/Lip;
    # W is used only by APG, not as the model-aggregation coefficient.
    w_current = project_rows_to_simplex(w_prev)
    w_previous = w_current.copy()
    k_current = 1.0
    objective_history = [
        objective_value(w_current, loss_normalized, time_normalized, class_mix, w_prev, parameters)
    ]
    completed = 0
    for iteration in range(parameters.max_iterations):
        k_next = (1.0 + np.sqrt(1.0 + 4.0 * k_current**2)) / 2.0
        extrapolation = (k_current - 1.0) / (k_current + 1.0)
        y_value = w_current + extrapolation * (w_current - w_previous)
        z_value = y_value - step_size * objective_gradient(
            y_value, loss_normalized, time_normalized, class_mix, w_prev, parameters
        )
        w_next = project_rows_to_simplex(z_value)
        value_next = objective_value(
            w_next, loss_normalized, time_normalized, class_mix, w_prev, parameters
        )
        value_current = objective_history[-1]
        objective_history.append(value_next)
        completed = iteration + 1
        w_previous, w_current = w_current, w_next
        k_current = k_next
        if abs(value_next - value_current) < parameters.psi:
            break

    adjacency, safeguard_hits = threshold_weights(w_current, parameters.epsilon)
    return APGResult(
        weights=w_current,
        adjacency=adjacency,
        objective=objective_history[-1],
        iterations=completed,
        safeguard_hits=safeguard_hits,
        objective_history=tuple(objective_history),
        lipschitz_constant=lipschitz,
        step_size=step_size,
    )