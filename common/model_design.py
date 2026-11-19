"""Validation and size estimation for user-designed MNIST image models."""

from __future__ import annotations

import math
from typing import Any


MAX_MODEL_PARAMETERS = 20_000_000
MAX_MODEL_WEIGHT_BYTES = 128 * 1024 * 1024
# Training-time data augmentation applied by the workers (no model layers, no weights):
# "flip_shift" = random horizontal flip + random shift of up to 4 pixels (pad and crop),
# the standard CIFAR-10 recipe. Not meant for digits (a flipped digit is another symbol).
AUGMENTATIONS = ("none", "flip_shift")
ACTIVATIONS = {
    "relu",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "linear",
    "softmax",
    "elu",
    "leaky_relu",
}
LAYER_TYPES = {
    "conv2d",
    "separable_conv2d",
    "max_pooling2d",
    "average_pooling2d",
    "batch_normalization",
    "dropout",
    "flatten",
    "global_average_pooling2d",
    "dense",
    "patch_embedding",
    "transformer_encoder",
    "global_average_pooling1d",
}
PRETRAINED_DEIT = "facebook/deit-tiny-patch16-224"


class ModelDesignError(ValueError):
    pass


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ModelDesignError(f"{label} must be an integer from {minimum} to {maximum}")
    return value


def validate_normalization(normalization: Any, channels: int) -> dict[str, Any]:
    if not isinstance(normalization, dict):
        raise ModelDesignError("Image normalization metadata must be an object")
    method = normalization.get("method")
    if method == "none":
        return {"method": "none"}
    if method == "minmax":
        return {"method": "minmax"}
    if method != "standard":
        raise ModelDesignError("Image normalization method must be standard, minmax, or none")
    mean = normalization.get("mean")
    standard_deviation = normalization.get("standard_deviation")
    if (
        not isinstance(mean, list)
        or not isinstance(standard_deviation, list)
        or len(mean) != channels
        or len(standard_deviation) != channels
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in [*mean, *standard_deviation]
        )
        or any(value <= 0 for value in standard_deviation)
    ):
        raise ModelDesignError("Standard normalization metadata is invalid for the image channels")
    return {
        "method": "standard",
        "mean": [float(value) for value in mean],
        "standard_deviation": [float(value) for value in standard_deviation],
    }


def validate_model_design(
    design: Any,
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> tuple[dict[str, Any], int]:
    if not isinstance(design, dict):
        raise ModelDesignError("Model design must be an object")
    training = design.get("training", {})
    if not isinstance(training, dict):
        raise ModelDesignError("Training settings must be an object")
    optimizer = training.get("optimizer", "adam")
    if not isinstance(optimizer, str) or optimizer not in {"adam", "adamw", "sgd", "rmsprop"}:
        raise ModelDesignError("Choose Adam, AdamW, SGD, or RMSprop")
    learning_rate = training.get("learning_rate", 0.001)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not 0.000001 <= learning_rate <= 0.1
    ):
        raise ModelDesignError("Learning rate must be between 0.000001 and 0.1")
    batch_size = _integer(training.get("batch_size", 32), "Batch size", 8, 256)
    normalized_training = {
        "optimizer": optimizer,
        "learning_rate": float(learning_rate),
        "batch_size": batch_size,
    }
    augmentation = training.get("augmentation", "none")
    if augmentation not in AUGMENTATIONS:
        raise ModelDesignError(f"Augmentation must be one of: {', '.join(AUGMENTATIONS)}")
    # Recorded only when enabled, so designs without augmentation keep their hashes.
    if augmentation != "none":
        normalized_training["augmentation"] = augmentation
    if (
        not isinstance(input_shape, (tuple, list))
        or len(input_shape) != 3
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or not 1 <= dimension <= 2048
            for dimension in input_shape
        )
        or input_shape[2] not in {1, 3}
    ):
        raise ModelDesignError("Input shape must be a supported H x W x C image shape")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or not 2 <= num_classes <= 1000:
        raise ModelDesignError("Number of output classes must be between 2 and 1000")
    if design.get("model_type") == "pretrained_deit":
        if design.get("model_id") != PRETRAINED_DEIT:
            raise ModelDesignError(f"Only the verified pretrained model {PRETRAINED_DEIT} is supported")
        # Replace the pretrained 1,000-class head with one sized for this dataset.
        return {
            "model_type": "pretrained_deit",
            "model_id": PRETRAINED_DEIT,
            "training": normalized_training,
        }, 5_717_416 - (1000 * 192 + 1000) + num_classes * 192 + num_classes

    raw_layers = design.get("layers")
    if design.get("model_type") != "custom" or not isinstance(raw_layers, list):
        raise ModelDesignError("Choose a custom architecture or the pretrained DeiT-Tiny model")
    if not 2 <= len(raw_layers) <= 32:
        raise ModelDesignError("Custom architectures must contain 2 to 32 layers")
    height, width, channels = input_shape
    shape: tuple[int, ...] = (height, width, channels)
    parameter_count = 0
    layers: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_layers):
        label = f"Layer {index + 1}"
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("type"), str)
            or raw["type"] not in LAYER_TYPES
        ):
            raise ModelDesignError(f"{label} has an unsupported layer type")
        layer = dict(raw)
        layer_type = layer["type"]
        if layer_type == "conv2d":
            if len(shape) != 3:
                raise ModelDesignError(f"{label} Conv2D requires an image or feature-map input")
            filters = _integer(layer.get("filters"), f"{label} filters", 1, 1024)
            kernel = _integer(layer.get("kernel_size"), f"{label} kernel size", 1, 15)
            stride = _integer(layer.get("strides", 1), f"{label} stride", 1, 8)
            padding = layer.get("padding", "same")
            activation = layer.get("activation", "relu")
            if (
                not isinstance(padding, str)
                or padding not in {"same", "valid"}
                or not isinstance(activation, str)
                or activation not in ACTIVATIONS
            ):
                raise ModelDesignError(f"{label} has an unsupported padding or activation")
            out_height = math.ceil(shape[0] / stride) if padding == "same" else max(0, (shape[0] - kernel) // stride + 1)
            out_width = math.ceil(shape[1] / stride) if padding == "same" else max(0, (shape[1] - kernel) // stride + 1)
            if not out_height or not out_width:
                raise ModelDesignError(f"{label} kernel is larger than its input")
            parameter_count += kernel * kernel * shape[2] * filters + filters
            shape = (out_height, out_width, filters)
            layer = {"type": layer_type, "filters": filters, "kernel_size": kernel, "strides": stride, "padding": padding, "activation": activation}
        elif layer_type == "separable_conv2d":
            if len(shape) != 3:
                raise ModelDesignError(f"{label} SeparableConv2D requires an image or feature-map input")
            filters = _integer(layer.get("filters"), f"{label} filters", 1, 1024)
            kernel = _integer(layer.get("kernel_size"), f"{label} kernel size", 1, 15)
            stride = _integer(layer.get("strides", 1), f"{label} stride", 1, 8)
            padding = layer.get("padding", "same")
            activation = layer.get("activation", "relu")
            if (
                not isinstance(padding, str)
                or padding not in {"same", "valid"}
                or not isinstance(activation, str)
                or activation not in ACTIVATIONS
            ):
                raise ModelDesignError(f"{label} has an unsupported padding or activation")
            out_height = math.ceil(shape[0] / stride) if padding == "same" else max(0, (shape[0] - kernel) // stride + 1)
            out_width = math.ceil(shape[1] / stride) if padding == "same" else max(0, (shape[1] - kernel) // stride + 1)
            if not out_height or not out_width:
                raise ModelDesignError(f"{label} kernel is larger than its input")
            parameter_count += kernel * kernel * shape[2] + shape[2] * filters + filters
            shape = (out_height, out_width, filters)
            layer = {"type": layer_type, "filters": filters, "kernel_size": kernel, "strides": stride, "padding": padding, "activation": activation}
        elif layer_type in {"max_pooling2d", "average_pooling2d"}:
            if len(shape) != 3:
                raise ModelDesignError(f"{label} pooling requires an image or feature-map input")
            pool = _integer(layer.get("pool_size", 2), f"{label} pool size", 2, 8)
            if shape[0] < pool or shape[1] < pool:
                raise ModelDesignError(f"{label} pool size is larger than its input")
            shape = (math.ceil(shape[0] / pool), math.ceil(shape[1] / pool), shape[2])
            layer = {"type": layer_type, "pool_size": pool}
        elif layer_type == "batch_normalization":
            if not shape:
                raise ModelDesignError(f"{label} cannot normalize a scalar")
            parameter_count += 4 * shape[-1]
            layer = {"type": layer_type}
        elif layer_type == "dropout":
            rate = layer.get("rate", 0.25)
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate < 1:
                raise ModelDesignError(f"{label} dropout rate must be in [0, 1)")
            layer = {"type": layer_type, "rate": float(rate)}
        elif layer_type == "flatten":
            if len(shape) < 2:
                raise ModelDesignError(f"{label} Flatten requires a multi-dimensional input")
            shape = (math.prod(shape),)
            layer = {"type": layer_type}
        elif layer_type == "global_average_pooling2d":
            if len(shape) != 3:
                raise ModelDesignError(f"{label} global average pooling requires a feature map")
            shape = (shape[-1],)
            layer = {"type": layer_type}
        elif layer_type == "dense":
            if not shape:
                raise ModelDesignError(f"{label} Dense requires a feature input")
            units = _integer(layer.get("units"), f"{label} units", 1, 8192)
            activation = layer.get("activation", "relu")
            if not isinstance(activation, str) or activation not in ACTIVATIONS:
                raise ModelDesignError(f"{label} has an unsupported activation")
            parameter_count += shape[-1] * units + units
            shape = (*shape[:-1], units)
            layer = {"type": layer_type, "units": units, "activation": activation}
        elif layer_type == "patch_embedding":
            if len(shape) != 3:
                raise ModelDesignError(f"{label} patch embedding requires image input")
            patch = _integer(layer.get("patch_size", 4), f"{label} patch size", 1, 14)
            embed_dim = _integer(layer.get("embed_dim", 64), f"{label} embedding size", 8, 1024)
            if shape[0] % patch or shape[1] % patch:
                raise ModelDesignError(f"{label} patch size must evenly divide the input dimensions")
            parameter_count += patch * patch * shape[2] * embed_dim + embed_dim
            parameter_count += (shape[0] // patch) * (shape[1] // patch) * embed_dim
            shape = ((shape[0] // patch) * (shape[1] // patch), embed_dim)
            layer = {"type": layer_type, "patch_size": patch, "embed_dim": embed_dim}
        elif layer_type == "transformer_encoder":
            if len(shape) != 2:
                raise ModelDesignError(f"{label} transformer block requires patch/token input")
            heads = _integer(layer.get("num_heads"), f"{label} attention heads", 1, 32)
            ff_dim = _integer(layer.get("ff_dim"), f"{label} feed-forward size", shape[-1], 4096)
            dropout = layer.get("dropout", 0.1)
            activation = layer.get("activation", "gelu")
            if shape[-1] % heads:
                raise ModelDesignError(f"{label} embedding size must be divisible by the number of heads")
            if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
                raise ModelDesignError(f"{label} dropout rate must be in [0, 1)")
            if not isinstance(activation, str) or activation not in ACTIVATIONS:
                raise ModelDesignError(f"{label} has an unsupported activation")
            embed_dim = shape[-1]
            parameter_count += 4 * embed_dim * embed_dim + 4 * embed_dim
            parameter_count += 2 * embed_dim * ff_dim + ff_dim + embed_dim
            parameter_count += 4 * embed_dim
            layer = {"type": layer_type, "num_heads": heads, "ff_dim": ff_dim, "dropout": float(dropout), "activation": activation}
            layer["embed_dim"] = shape[-1]
        elif layer_type == "global_average_pooling1d":
            if len(shape) != 2:
                raise ModelDesignError(f"{label} token pooling requires sequence input")
            shape = (shape[-1],)
            layer = {"type": layer_type}
        layers.append(layer)
        if parameter_count > MAX_MODEL_PARAMETERS:
            raise ModelDesignError(
                f"Architecture has more than {MAX_MODEL_PARAMETERS:,} parameters"
            )
    if len(shape) != 1 or shape[0] != num_classes:
        raise ModelDesignError("Finish the model with a Dense layer whose units match the dataset classes")
    if layers[-1]["type"] != "dense" or layers[-1]["activation"] != "linear":
        raise ModelDesignError("The final layer must be a linear Dense classifier")
    if parameter_count < 1:
        raise ModelDesignError("The architecture has no trainable parameters")
    return {
        "model_type": "custom",
        "layers": layers,
        "training": normalized_training,
    }, parameter_count
