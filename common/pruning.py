"""Structured (channel / unit) pruning of a trained custom dataset model.

The model is the ``custom`` layer list understood by ``common.model_design``; its
weights are the flat Keras ``get_weights()`` list of the Sequential model the
workers build, so per layer:

* conv2d            -> kernel (k, k, in, out), bias (out,)
* separable_conv2d  -> depthwise (k, k, in, 1), pointwise (1, 1, in, out), bias (out,)
* batch_normalization -> gamma, beta, moving_mean, moving_variance (channels,)
* dense             -> kernel (in, out), bias (out,)
* pooling / dropout / flatten / global_average_pooling2d -> no weights

Each prunable layer (every conv/separable/dense layer except the final classifier)
keeps its most important output units. Importance is the L1 norm of the unit's
kernel slice, scaled by |gamma| when a batch-normalization layer follows (the
"network slimming" signal). Removing a unit also removes its batch-norm entries
and the matching input slice of the next weighted layer, so the result is a
smaller dense network — no sparse masks — that runs faster on a Jetson.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from common.model_design import ModelDesignError, validate_model_design

PRUNABLE_LAYERS = {"conv2d", "separable_conv2d", "dense"}
PASS_THROUGH_LAYERS = {
    "batch_normalization",
    "max_pooling2d",
    "average_pooling2d",
    "dropout",
    "flatten",
    "global_average_pooling2d",
}
WEIGHT_COUNTS = {"conv2d": 2, "separable_conv2d": 3, "dense": 2, "batch_normalization": 4}
MIN_KEPT_UNITS = 4
MAX_PRUNE_RATIO = 0.8
# Descriptive NAS keys that no longer describe a pruned network.
_STALE_KEYS = ("blocks", "channels", "channel_schedule", "block_type", "kernel_size", "head_units", "dropout", "attention", "convs_per_block")


class PruningError(ValueError):
    """The model cannot be pruned (unsupported layers or inconsistent weights)."""


def _split_weights(layers: list[dict[str, Any]], weights: list[np.ndarray]) -> list[list[np.ndarray]]:
    groups: list[list[np.ndarray]] = []
    cursor = 0
    for layer in layers:
        count = WEIGHT_COUNTS.get(layer["type"], 0)
        groups.append([np.asarray(value) for value in weights[cursor:cursor + count]])
        cursor += count
    if cursor != len(weights):
        raise PruningError("The weight list does not match the architecture layers")
    return groups


def _spatial_shapes(layers: list[dict[str, Any]], input_shape: tuple[int, int, int]) -> list[tuple[int, ...]]:
    """Output shape of every layer (needed to map channels through Flatten)."""
    shape: tuple[int, ...] = tuple(input_shape)
    shapes = []
    for layer in layers:
        kind = layer["type"]
        if kind in {"conv2d", "separable_conv2d"}:
            stride, kernel = layer["strides"], layer["kernel_size"]
            if layer["padding"] == "same":
                height, width = math.ceil(shape[0] / stride), math.ceil(shape[1] / stride)
            else:
                height, width = (shape[0] - kernel) // stride + 1, (shape[1] - kernel) // stride + 1
            shape = (height, width, layer["filters"])
        elif kind in {"max_pooling2d", "average_pooling2d"}:
            pool = layer["pool_size"]
            shape = (math.ceil(shape[0] / pool), math.ceil(shape[1] / pool), shape[2])
        elif kind == "flatten":
            shape = (math.prod(shape),)
        elif kind == "global_average_pooling2d":
            shape = (shape[-1],)
        elif kind == "dense":
            shape = (*shape[:-1], layer["units"])
        shapes.append(shape)
    return shapes


def _importance(kind: str, params: list[np.ndarray], following_gamma: np.ndarray | None) -> np.ndarray:
    if kind == "conv2d":
        scores = np.abs(params[0]).sum(axis=(0, 1, 2))
    elif kind == "separable_conv2d":
        scores = np.abs(params[1]).sum(axis=(0, 1, 2))
    else:
        scores = np.abs(params[0]).sum(axis=0)
    if following_gamma is not None:
        scores = scores * np.abs(following_gamma)
    return scores.astype(np.float64)


def prune_model(
    architecture: dict[str, Any],
    weights: list[np.ndarray],
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    ratio: float,
) -> tuple[dict[str, Any], list[np.ndarray], dict[str, Any]]:
    """Remove ``ratio`` of the units of every hidden layer; return (design, weights, report)."""
    if not 0 < ratio <= MAX_PRUNE_RATIO:
        raise PruningError(f"Pruning ratio must be in (0, {MAX_PRUNE_RATIO}]")
    try:
        design, parameters_before = validate_model_design(architecture, input_shape, num_classes)
    except ModelDesignError as error:
        raise PruningError(str(error)) from error
    if design["model_type"] != "custom":
        raise PruningError("Only custom layer models can be pruned")
    layers = [dict(layer) for layer in design["layers"]]
    unsupported = sorted({layer["type"] for layer in layers} - PRUNABLE_LAYERS - PASS_THROUGH_LAYERS)
    if unsupported:
        raise PruningError(f"Structured pruning does not support layer types: {', '.join(unsupported)}")
    groups = _split_weights(layers, weights)
    shapes = _spatial_shapes(layers, input_shape)
    classifier = max(index for index, layer in enumerate(layers) if layer["type"] == "dense")

    report_layers = []
    for index, layer in enumerate(layers):
        kind = layer["type"]
        if kind not in PRUNABLE_LAYERS or index == classifier:
            continue
        params = groups[index]
        units = params[-1].shape[0]
        # A batch-norm directly after the layer scales each unit by gamma.
        follower = index + 1 if index + 1 < len(layers) and layers[index + 1]["type"] == "batch_normalization" else None
        gamma = groups[follower][0] if follower is not None else None
        keep_count = max(min(MIN_KEPT_UNITS, units), int(round(units * (1 - ratio))))
        if keep_count >= units:
            continue
        scores = _importance(kind, params, gamma)
        keep = np.sort(np.argsort(-scores, kind="stable")[:keep_count])

        # Slice this layer's outputs.
        if kind == "conv2d":
            groups[index] = [params[0][..., keep], params[1][keep]]
            layer["filters"] = keep_count
        elif kind == "separable_conv2d":
            groups[index] = [params[0], params[1][..., keep], params[2][keep]]
            layer["filters"] = keep_count
        else:
            groups[index] = [params[0][:, keep], params[1][keep]]
            layer["units"] = keep_count
        if follower is not None:
            groups[follower] = [value[keep] for value in groups[follower]]

        # Slice the input of the next weighted (non-batch-norm) layer.
        cursor = index + 1
        flatten_shape: tuple[int, ...] | None = None
        while cursor < len(layers) and layers[cursor]["type"] not in PRUNABLE_LAYERS:
            if layers[cursor]["type"] == "flatten":
                flatten_shape = shapes[cursor - 1]
            cursor += 1
        if cursor >= len(layers):
            raise PruningError("A pruned layer has no following weighted layer")
        nxt = groups[cursor]
        next_kind = layers[cursor]["type"]
        if next_kind == "conv2d":
            groups[cursor] = [nxt[0][:, :, keep, :], nxt[1]]
        elif next_kind == "separable_conv2d":
            groups[cursor] = [nxt[0][:, :, keep, :], nxt[1][:, :, keep, :], nxt[2]]
        elif flatten_shape is not None:
            # Flatten of (h, w, c) is row-major: index = position * c + channel.
            channels = flatten_shape[-1]
            positions = math.prod(flatten_shape[:-1])
            rows = (np.arange(positions)[:, None] * channels + keep[None, :]).reshape(-1)
            groups[cursor] = [nxt[0][rows, :], nxt[1]]
        else:
            groups[cursor] = [nxt[0][keep, :], nxt[1]]
        shapes = _spatial_shapes(layers, input_shape)
        report_layers.append({"layer": index, "type": kind, "units_before": units, "units_after": keep_count})

    if not report_layers:
        raise PruningError("No layer is large enough to prune at this ratio")
    pruned = {
        **{key: value for key, value in architecture.items() if key not in _STALE_KEYS},
        "layers": layers,
        "training": design["training"],
    }
    base_id = str(architecture.get("id") or "model")
    pruned["id"] = f"{base_id}-pruned{int(round(ratio * 100))}"[:128]
    normalized, parameters_after = validate_model_design(pruned, input_shape, num_classes)
    pruned["parameter_count"] = parameters_after
    flat = [value.astype(np.float32) for group in groups for value in group]
    if sum(int(value.size) for value in flat) != parameters_after:
        raise PruningError("Pruned weights do not match the pruned architecture")
    report = {
        "ratio": ratio,
        "parameters_before": parameters_before,
        "parameters_after": parameters_after,
        "compression": round(parameters_before / parameters_after, 3),
        "layers": report_layers,
    }
    return pruned, flat, report
