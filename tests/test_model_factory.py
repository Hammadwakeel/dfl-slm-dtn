import sys
from types import SimpleNamespace

import numpy as np

from worker.model_factory import build_dataset_model


def test_pretrained_deit_grayscale_standard_normalization_preserves_channels(
    monkeypatch,
):
    import tensorflow as tf

    class FakeViTModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, pixel_values, **_kwargs):
            pooled = tf.reduce_mean(pixel_values, axis=(2, 3))
            return SimpleNamespace(last_hidden_state=pooled[:, tf.newaxis, :])

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TFViTModel=FakeViTModel),
    )
    model = build_dataset_model(
        input_shape=(28, 28, 1),
        num_classes=10,
        architecture={
            "model_type": "pretrained_deit",
            "model_id": "facebook/deit-tiny-patch16-224",
            "normalization": {
                "method": "standard",
                "mean": [0.13],
                "standard_deviation": [0.3],
            },
        },
        seed=42,
    )

    predictions = model(np.zeros((2, 28, 28, 1), dtype=np.float32), training=False)

    assert tuple(predictions.shape) == (2, 10)
    assert np.isfinite(predictions.numpy()).all()
