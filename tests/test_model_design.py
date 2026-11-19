import pytest

from common.model_design import (
    MAX_MODEL_PARAMETERS,
    ModelDesignError,
    PRETRAINED_DEIT,
    validate_model_design,
    validate_normalization,
)


def test_custom_cnn_design_is_normalized_and_parameter_counted():
    design, parameter_count = validate_model_design(
        {
            "model_type": "custom",
            "layers": [
                {"type": "conv2d", "filters": 8, "kernel_size": 3},
                {"type": "global_average_pooling2d"},
                {"type": "dense", "units": 10, "activation": "linear"},
            ],
        },
        (28, 28, 1),
        10,
    )

    assert design["training"] == {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 32,
    }
    assert design["layers"][0] == {
        "type": "conv2d",
        "filters": 8,
        "kernel_size": 3,
        "strides": 1,
        "padding": "same",
        "activation": "relu",
    }
    assert parameter_count == 170


def test_custom_transformer_design_counts_patch_attention_and_classifier_weights():
    _, parameter_count = validate_model_design(
        {
            "model_type": "custom",
            "layers": [
                {"type": "patch_embedding", "patch_size": 7, "embed_dim": 32},
                {"type": "transformer_encoder", "num_heads": 4, "ff_dim": 64},
                {"type": "global_average_pooling1d"},
                {"type": "dense", "units": 10, "activation": "linear"},
            ],
        },
        (28, 28, 1),
        10,
    )

    assert parameter_count == 10_986
    assert parameter_count < MAX_MODEL_PARAMETERS


def test_custom_design_rejects_bad_classifier_and_parameter_overflow():
    with pytest.raises(ModelDesignError, match="linear Dense"):
        validate_model_design(
            {
                "model_type": "custom",
                "layers": [
                    {"type": "flatten"},
                    {"type": "dense", "units": 10, "activation": "softmax"},
                ],
            },
            (28, 28, 1),
            10,
        )
    with pytest.raises(ModelDesignError, match="more than"):
        validate_model_design(
            {
                "model_type": "custom",
                "layers": [
                    {"type": "flatten"},
                    {"type": "dense", "units": 8192},
                    {"type": "dense", "units": 8192},
                    {"type": "dense", "units": 10, "activation": "linear"},
                ],
            },
            (28, 28, 1),
            10,
        )


def test_pretrained_deit_and_normalization_are_restricted_and_validated():
    design, parameter_count = validate_model_design(
        {"model_type": "pretrained_deit", "model_id": PRETRAINED_DEIT},
        (28, 28, 1),
        10,
    )
    assert design["model_id"] == PRETRAINED_DEIT
    assert parameter_count == 5_526_346
    assert parameter_count < MAX_MODEL_PARAMETERS
    assert validate_normalization(
        {"method": "standard", "mean": [0.1], "standard_deviation": [0.2]},
        1,
    ) == {
        "method": "standard",
        "mean": [0.1],
        "standard_deviation": [0.2],
    }
    with pytest.raises(ModelDesignError, match="normalization metadata"):
        validate_normalization(
            {"method": "standard", "mean": [0.1], "standard_deviation": [0]},
            1,
        )


def test_augmentation_is_recorded_only_when_enabled_and_validated():
    layers = [
        {"type": "conv2d", "filters": 8, "kernel_size": 3},
        {"type": "global_average_pooling2d"},
        {"type": "dense", "units": 10, "activation": "linear"},
    ]
    plain, _ = validate_model_design({"model_type": "custom", "layers": layers}, (32, 32, 3), 10)
    assert "augmentation" not in plain["training"]
    none, _ = validate_model_design(
        {"model_type": "custom", "layers": layers, "training": {"augmentation": "none"}},
        (32, 32, 3),
        10,
    )
    assert none == plain  # explicit "none" keeps the design (and its hash) unchanged
    augmented, _ = validate_model_design(
        {"model_type": "custom", "layers": layers, "training": {"augmentation": "flip_shift"}},
        (32, 32, 3),
        10,
    )
    assert augmented["training"]["augmentation"] == "flip_shift"
    with pytest.raises(ModelDesignError, match="Augmentation"):
        validate_model_design(
            {"model_type": "custom", "layers": layers, "training": {"augmentation": "mixup"}},
            (32, 32, 3),
            10,
        )
