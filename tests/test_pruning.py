import numpy as np
import pytest

from common.pruning import PruningError, prune_model


def _relu(value: np.ndarray) -> np.ndarray:
    return np.maximum(value, 0)


def _conv_1x1(images: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.einsum("nhwc,cf->nhwf", images, kernel[0, 0]) + bias


def _batch_norm(values: np.ndarray, gamma, beta, mean, variance) -> np.ndarray:
    return gamma * (values - mean) / np.sqrt(variance + 1e-3) + beta


def _forward_conv_gap(weights: list[np.ndarray], images: np.ndarray) -> np.ndarray:
    kernel, bias, gamma, beta, mean, variance, dense_kernel, dense_bias = weights
    features = _batch_norm(_relu(_conv_1x1(images, kernel, bias)), gamma, beta, mean, variance)
    return features.mean(axis=(1, 2)) @ dense_kernel + dense_bias


def _forward_conv_flatten(weights: list[np.ndarray], images: np.ndarray) -> np.ndarray:
    kernel, bias, hidden_kernel, hidden_bias, out_kernel, out_bias = weights
    features = _relu(_conv_1x1(images, kernel, bias)).reshape(images.shape[0], -1)
    return _relu(features @ hidden_kernel + hidden_bias) @ out_kernel + out_bias


CONV_GAP = {
    "model_type": "custom",
    "id": "nas-test",
    "channels": 8,
    "layers": [
        {"type": "conv2d", "filters": 8, "kernel_size": 1, "padding": "same", "activation": "relu"},
        {"type": "batch_normalization"},
        {"type": "global_average_pooling2d"},
        {"type": "dense", "units": 3, "activation": "linear"},
    ],
    "training": {"optimizer": "adam", "learning_rate": 0.001, "batch_size": 32},
}

CONV_FLATTEN = {
    "model_type": "custom",
    "layers": [
        {"type": "conv2d", "filters": 6, "kernel_size": 1, "padding": "same", "activation": "relu"},
        {"type": "flatten"},
        {"type": "dense", "units": 10, "activation": "relu"},
        {"type": "dense", "units": 3, "activation": "linear"},
    ],
}


def test_pruning_dead_units_preserves_outputs_through_batch_norm_and_gap() -> None:
    rng = np.random.default_rng(0)
    kernel = rng.normal(size=(1, 1, 3, 8)).astype(np.float32)
    bias = rng.normal(size=8).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=8).astype(np.float32)
    beta = np.zeros(8, np.float32)
    dead = [1, 4, 6, 7]
    kernel[..., dead] = 0
    bias[dead] = 0
    gamma[dead] = 0
    weights = [kernel, bias, gamma, beta, np.zeros(8, np.float32), np.ones(8, np.float32),
               rng.normal(size=(8, 3)).astype(np.float32), rng.normal(size=3).astype(np.float32)]
    images = rng.normal(size=(5, 4, 4, 3)).astype(np.float32)

    design, pruned, report = prune_model(CONV_GAP, weights, input_shape=(4, 4, 3), num_classes=3, ratio=0.5)

    assert design["layers"][0]["filters"] == 4
    assert design["id"] == "nas-test-pruned50"
    assert "channels" not in design
    assert report["parameters_after"] < report["parameters_before"]
    assert [value.shape for value in pruned][:2] == [(1, 1, 3, 4), (4,)]
    np.testing.assert_allclose(_forward_conv_gap(pruned, images), _forward_conv_gap(weights, images), atol=1e-5)


def test_pruning_maps_channels_through_flatten_and_hidden_dense() -> None:
    rng = np.random.default_rng(1)
    kernel = rng.normal(size=(1, 1, 1, 6)).astype(np.float32)
    bias = rng.normal(size=6).astype(np.float32)
    kernel[..., [0, 3, 5]] = 0
    bias[[0, 3, 5]] = -1.0  # relu output is exactly zero for these channels
    hidden_kernel = rng.normal(size=(2 * 2 * 6, 10)).astype(np.float32)
    hidden_bias = rng.normal(size=10).astype(np.float32)
    dead_hidden = [1, 2, 7, 8, 9]
    hidden_kernel[:, dead_hidden] = 0
    hidden_bias[dead_hidden] = -1.0  # never active after relu
    out_kernel = rng.normal(size=(10, 3)).astype(np.float32)
    weights = [kernel, bias, hidden_kernel, hidden_bias, out_kernel, rng.normal(size=3).astype(np.float32)]
    images = rng.normal(size=(4, 2, 2, 1)).astype(np.float32)

    design, pruned, _ = prune_model(CONV_FLATTEN, weights, input_shape=(2, 2, 1), num_classes=3, ratio=0.5)

    assert design["layers"][0]["filters"] == 4  # min kept units for a 6-filter layer at 50% is 3 -> floor 4
    assert design["layers"][2]["units"] == 5
    assert pruned[2].shape == (2 * 2 * 4, 5)
    assert design["parameter_count"] == sum(value.size for value in pruned)
    np.testing.assert_allclose(_forward_conv_flatten(pruned, images), _forward_conv_flatten(weights, images), atol=1e-5)


def test_pruning_rejects_transformer_models_and_bad_ratios() -> None:
    transformer = {
        "model_type": "custom",
        "layers": [
            {"type": "patch_embedding", "patch_size": 2, "embed_dim": 8},
            {"type": "transformer_encoder", "num_heads": 2, "ff_dim": 16},
            {"type": "global_average_pooling1d"},
            {"type": "dense", "units": 3, "activation": "linear"},
        ],
    }
    with pytest.raises(PruningError, match="does not support"):
        prune_model(transformer, [], input_shape=(4, 4, 1), num_classes=3, ratio=0.5)
    with pytest.raises(PruningError, match="ratio"):
        prune_model(CONV_GAP, [], input_shape=(4, 4, 3), num_classes=3, ratio=0.95)
    with pytest.raises(PruningError, match="weight list"):
        prune_model(CONV_GAP, [np.zeros(1)], input_shape=(4, 4, 3), num_classes=3, ratio=0.5)
