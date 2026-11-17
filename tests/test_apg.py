import numpy as np

from topology_opt.apg import (
    APGParameters,
    adjacency_edges,
    count_edge_changes,
    map_real_worker_sender_edges,
    mesh_adjacency,
    normalize_off_diagonal,
    objective_gradient,
    objective_value,
    optimize_topology,
    project_rows_to_simplex,
    ring_adjacency,
    threshold_weights,
    uniform_mesh_weights,
)


def _one_hot_theta(node_count: int = 8) -> np.ndarray:
    theta = np.zeros((node_count, 10), dtype=np.float64)
    for node in range(node_count):
        theta[node, node % 10] = 1.0
    return theta


def _ring_weights(node_count: int = 8) -> np.ndarray:
    weights = np.zeros((node_count, node_count), dtype=np.float64)
    for receiver in range(node_count):
        weights[receiver, (receiver - 1) % node_count] = 1.0
    return weights


def test_projection_has_simplex_rows_zero_diagonal_and_is_idempotent() -> None:
    values = np.arange(64, dtype=np.float64).reshape(8, 8) / 7.0 - 4.0
    projected = project_rows_to_simplex(values)
    np.testing.assert_allclose(projected.sum(axis=1), np.ones(8), atol=1e-12)
    assert np.all(projected >= 0.0) and np.all(projected <= 1.0)
    np.testing.assert_array_equal(np.diag(projected), np.zeros(8))
    np.testing.assert_allclose(project_rows_to_simplex(projected), projected, atol=1e-12)


def test_normalization_ignores_nan_diagonal_but_rejects_nonfinite_off_diagonal() -> None:
    matrix = np.array(
        [[np.nan, 2.0, 4.0], [1.0, np.nan, 3.0], [4.0, 2.0, np.nan]],
        dtype=np.float64,
    )
    normalized = normalize_off_diagonal(matrix)
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(np.diag(normalized), np.zeros(3))
    assert normalized[1, 0] == 0.0
    assert normalized[0, 2] == 1.0

    matrix[0, 1] = np.nan
    try:
        normalize_off_diagonal(matrix)
    except ValueError as error:
        assert "off-diagonal inputs must be finite" in str(error)
    else:
        raise AssertionError("Non-finite off-diagonal input should be rejected")


def test_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(2026)
    n, k = 4, 3
    theta = rng.random((n, k))
    theta /= theta.sum(axis=1, keepdims=True)
    previous = project_rows_to_simplex(rng.normal(size=(n, n)))
    point = project_rows_to_simplex(rng.normal(size=(n, n)))
    loss = rng.random((n, n))
    wait = rng.random((n, n))
    parameters = APGParameters()
    analytical = objective_gradient(point, loss, wait, theta, previous, parameters)
    numerical = np.zeros_like(point)
    epsilon = 1e-6
    for row in range(n):
        for column in range(n):
            plus = point.copy()
            minus = point.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical[row, column] = (
                objective_value(plus, loss, wait, theta, previous, parameters)
                - objective_value(minus, loss, wait, theta, previous, parameters)
            ) / (2 * epsilon)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-6)


def test_apg_is_deterministic_for_fixed_inputs() -> None:
    rng = np.random.default_rng(7)
    theta = _one_hot_theta()
    loss = rng.random((8, 8))
    wait = rng.random((8, 8))
    np.fill_diagonal(loss, 0.0)
    np.fill_diagonal(wait, 0.0)
    previous = uniform_mesh_weights(8)
    first = optimize_topology(loss, wait, theta, previous, mesh_adjacency(8))
    second = optimize_topology(loss, wait, theta, previous, mesh_adjacency(8))
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.adjacency, second.adjacency)
    assert first.objective_history == second.objective_history


def test_apg_objective_converges_on_a_convex_case_with_documented_tolerance() -> None:
    rng = np.random.default_rng(11)
    theta = _one_hot_theta()
    previous = _ring_weights()
    loss = rng.random((8, 8))
    wait = rng.random((8, 8))
    np.fill_diagonal(loss, 0.0)
    np.fill_diagonal(wait, 0.0)
    result = optimize_topology(loss, wait, theta, previous, ring_adjacency(8))
    assert np.isfinite(result.objective_history).all()
    assert result.objective <= result.objective_history[0] + 1e-10
    # Nesterov's specified extrapolation can be non-monotone; allow 0.02 absolute transient increase.
    assert max(result.objective_history) <= result.objective_history[0] + 0.02


def test_constant_inputs_at_optimum_keep_topology_unchanged() -> None:
    previous = _ring_weights()
    theta = np.full((8, 10), 0.1, dtype=np.float64)
    loss = np.full((8, 8), 0.5, dtype=np.float64)
    wait = np.full((8, 8), 1.0, dtype=np.float64)
    np.fill_diagonal(loss, 0.0)
    np.fill_diagonal(wait, 0.0)
    prior_adjacency = ring_adjacency(8)
    result = optimize_topology(loss, wait, theta, previous, prior_adjacency)
    np.testing.assert_allclose(result.weights, previous, atol=1e-12)
    np.testing.assert_array_equal(result.adjacency, prior_adjacency)
    assert count_edge_changes(prior_adjacency, result.adjacency) == 0


def test_changing_loss_time_or_theta_can_change_selected_edges() -> None:
    n = 8
    previous = np.zeros((n, n), dtype=np.float64)
    for receiver in range(n):
        group_start = 0 if receiver < 4 else 4
        group_end = group_start + 4
        same_group = [node for node in range(group_start, group_end) if node != receiver]
        previous[receiver, same_group[0]] = 1.0
    prior_adjacency = (previous > 0).astype(np.int8)
    uniform_theta = np.full((n, 2), 0.5, dtype=np.float64)
    constant = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(constant, 0.0)
    baseline = optimize_topology(constant, constant, uniform_theta, previous, prior_adjacency).adjacency

    loss = constant.copy()
    for receiver in range(n):
           loss[receiver, (receiver - 1) % n] = 5.0
           loss[receiver, (receiver + 1) % n] = 0.0
    changed_loss = optimize_topology(loss, constant, uniform_theta, previous, prior_adjacency).adjacency

    wait = constant.copy()
    for receiver in range(n):
        wait[receiver, (receiver - 1) % n] = 5.0
        wait[receiver, (receiver + 1) % n] = 0.0
    changed_wait = optimize_topology(constant, wait, uniform_theta, previous, prior_adjacency).adjacency

    theta_previous = _ring_weights(n)
    theta_previous[0] = 0.0
    theta_previous[0, 1] = 0.15
    theta_previous[0, 2] = 0.85
    theta_prior_adjacency = ring_adjacency(n)
    theta_params = APGParameters(psi=0.0)
    theta_baseline = optimize_topology(
        constant, constant, uniform_theta, theta_previous, theta_prior_adjacency, theta_params
    ).adjacency
    nonuniform_theta = np.zeros((n, 2), dtype=np.float64)
    nonuniform_theta[0] = [0.5, 0.5]
    nonuniform_theta[1] = [1.0, 0.0]
    nonuniform_theta[2] = [0.5, 0.5]
    nonuniform_theta[3:] = [0.4, 0.6]
    changed_theta = optimize_topology(
        constant, constant, nonuniform_theta, theta_previous, theta_prior_adjacency, theta_params
    ).adjacency

    assert count_edge_changes(baseline, changed_loss) > 0
    assert count_edge_changes(baseline, changed_wait) > 0
    assert count_edge_changes(theta_baseline, changed_theta) > 0


def test_threshold_and_empty_row_safeguard_are_deterministic() -> None:
    weights = uniform_mesh_weights(8)
    adjacency, safeguard_hits = threshold_weights(weights, epsilon=0.15)
    assert safeguard_hits == 8
    assert int(adjacency.sum()) == 8
    np.testing.assert_array_equal(np.diag(adjacency), np.zeros(8, dtype=np.int8))
    assert adjacency_edges(adjacency) == [(row, 0 if row else 1) for row in range(8)]


def test_real_worker_ring_sender_mapping_matches_node_5_to_node_7_map() -> None:
    assert map_real_worker_sender_edges("ring") == {
        ("node_5", "node_7"),
        ("node_6", "node_5"),
        ("node_7", "node_6"),
    }
    assert map_real_worker_sender_edges("mesh") == {
        (f"node_{receiver}", f"node_{sender}")
        for receiver in range(5, 8)
        for sender in range(5, 8)
        if receiver != sender
    }