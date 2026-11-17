import numpy as np
import pytest

from worker.aggregation import Contribution, aggregate, mixing_weights, model_signature


def _model(value: float, node: str = "w", samples: int = 1, degree: int = 1) -> Contribution:
    return Contribution(node, [np.full((2,), value, dtype=np.float32), np.full((1,), value, dtype=np.float32)], samples, degree)


def test_fedavg_weights_by_sample_count():
    result = aggregate("fedavg", [_model(1.0, samples=2), _model(4.0, samples=1)])
    np.testing.assert_allclose(result[0], [2.0, 2.0])
    assert result[0].dtype == np.float32


def test_uniform_ignores_sample_counts():
    result = aggregate("uniform", [_model(1.0, samples=9), _model(3.0, samples=1)])
    np.testing.assert_allclose(result[0], [2.0, 2.0])


def test_metropolis_weights_sum_to_one_and_favor_low_degree_links():
    contributions = [_model(0.0, "w1", degree=2), _model(3.0, "w2", degree=2), _model(6.0, "w3", degree=4)]
    coefficients = mixing_weights("metropolis", contributions)
    assert sum(coefficients) == pytest.approx(1.0)
    assert coefficients[1] == pytest.approx(1 / 3)
    assert coefficients[2] == pytest.approx(1 / 5)
    result = aggregate("metropolis", contributions)
    np.testing.assert_allclose(result[0], [3.0 / 3 + 6.0 / 5] * 2, rtol=1e-6)


def test_trimmed_mean_drops_an_outlying_peer_and_falls_back_with_two_models():
    result = aggregate("trimmed_mean", [_model(1.0), _model(2.0), _model(1000.0)])
    np.testing.assert_allclose(result[0], [2.0, 2.0])
    fallback = aggregate("trimmed_mean", [_model(1.0, samples=1), _model(3.0, samples=1)])
    np.testing.assert_allclose(fallback[0], [2.0, 2.0])


def test_architecture_mismatch_names_the_peer():
    bad = Contribution("worker4", [np.zeros((3,), dtype=np.float32), np.zeros((1,), dtype=np.float32)], 1, 1)
    with pytest.raises(ValueError, match="worker4.*architecture mismatch"):
        aggregate("fedavg", [_model(1.0), bad])
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        aggregate("median", [_model(1.0)])


def test_model_signature_depends_on_layout_not_values():
    assert model_signature(_model(1.0).weights) == model_signature(_model(5.0).weights)
    assert model_signature(_model(1.0).weights) != model_signature([np.zeros((3,), dtype=np.float32)])
