"""TensorFlow model construction for custom and pretrained MNIST architectures."""

from __future__ import annotations

from typing import Any


class TransformerEncoderBlock:
    def __new__(
        cls,
        tf: Any,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        activation: str,
    ) -> Any:
        class EncoderBlock(tf.keras.layers.Layer):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.attention = tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embed_dim // num_heads,
                    dropout=dropout,
                )
                self.drop1 = tf.keras.layers.Dropout(dropout)
                self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.ff1 = tf.keras.layers.Dense(ff_dim, activation=activation)
                self.ff2 = tf.keras.layers.Dense(embed_dim)
                self.drop2 = tf.keras.layers.Dropout(dropout)

            def call(self, inputs: Any, training: bool = False) -> Any:
                normalized = self.norm1(inputs)
                attended = self.attention(normalized, normalized, training=training)
                values = inputs + self.drop1(attended, training=training)
                normalized = self.norm2(values)
                feed_forward = self.ff2(self.ff1(normalized))
                return values + self.drop2(feed_forward, training=training)

        return EncoderBlock()


def build_dataset_model(
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    architecture: dict[str, Any],
    seed: int,
) -> Any:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    if architecture.get("model_type") == "pretrained_deit":
        return _build_pretrained_deit(tf, input_shape, num_classes, architecture)

    model_layers: list[Any] = [tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)]
    for spec in architecture["layers"]:
        layer_type = spec["type"]
        if layer_type == "conv2d":
            model_layers.append(
                tf.keras.layers.Conv2D(
                    filters=spec["filters"],
                    kernel_size=spec["kernel_size"],
                    strides=spec["strides"],
                    padding=spec["padding"],
                    activation=spec["activation"],
                )
            )
        elif layer_type == "separable_conv2d":
            model_layers.append(
                tf.keras.layers.SeparableConv2D(
                    filters=spec["filters"],
                    kernel_size=spec["kernel_size"],
                    strides=spec["strides"],
                    padding=spec["padding"],
                    activation=spec["activation"],
                )
            )
        elif layer_type == "max_pooling2d":
            model_layers.append(tf.keras.layers.MaxPooling2D(pool_size=spec["pool_size"], padding="same"))
        elif layer_type == "average_pooling2d":
            model_layers.append(tf.keras.layers.AveragePooling2D(pool_size=spec["pool_size"], padding="same"))
        elif layer_type == "batch_normalization":
            model_layers.append(tf.keras.layers.BatchNormalization())
        elif layer_type == "dropout":
            model_layers.append(tf.keras.layers.Dropout(rate=spec["rate"]))
        elif layer_type == "flatten":
            model_layers.append(tf.keras.layers.Flatten())
        elif layer_type == "global_average_pooling2d":
            model_layers.append(tf.keras.layers.GlobalAveragePooling2D())
        elif layer_type == "dense":
            model_layers.append(
                tf.keras.layers.Dense(
                    units=spec["units"],
                    activation=None if spec["activation"] == "linear" else spec["activation"],
                )
            )
        elif layer_type == "patch_embedding":
            model_layers.append(
                tf.keras.layers.Conv2D(
                    filters=spec["embed_dim"],
                    kernel_size=spec["patch_size"],
                    strides=spec["patch_size"],
                    padding="valid",
                )
            )
            model_layers.append(tf.keras.layers.Reshape((-1, spec["embed_dim"])))
            model_layers.append(_position_embedding_layer(tf, name=f"position_embedding_{len(model_layers)}"))
        elif layer_type == "transformer_encoder":
            model_layers.append(
                TransformerEncoderBlock(
                    tf,
                    spec["embed_dim"],
                    spec["num_heads"],
                    spec["ff_dim"],
                    spec["dropout"],
                    spec["activation"],
                )
            )
        elif layer_type == "global_average_pooling1d":
            model_layers.append(tf.keras.layers.GlobalAveragePooling1D())
        else:
            raise ValueError(f"Unsupported model layer type: {layer_type}")

    return tf.keras.Sequential(model_layers, name="custom_mnist_model")


def _position_embedding_layer(tf: Any, **kwargs: Any) -> Any:
    class PositionEmbeddingLayer(tf.keras.layers.Layer):
        def build(self, input_shape: Any) -> None:
            self.position_embeddings = self.add_weight(
                name="position_embeddings",
                shape=(int(input_shape[1]), int(input_shape[2])),
                initializer="zeros",
                trainable=True,
            )

        def call(self, inputs: Any) -> Any:
            return inputs + self.position_embeddings

    return PositionEmbeddingLayer(**kwargs)


def _build_pretrained_deit(
    tf: Any,
    input_shape: tuple[int, int, int],
    num_classes: int,
    architecture: dict[str, Any],
) -> Any:
    try:
        from transformers import TFViTModel
    except ImportError as error:
        raise RuntimeError(
            "Pretrained DeiT-Tiny requires the worker dependency 'transformers==4.44.2'"
        ) from error

    backbone = TFViTModel.from_pretrained(
        architecture["model_id"],
        add_pooling_layer=False,
        trust_remote_code=False,
    )
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, name="mnist_image")
    images = tf.keras.layers.Resizing(224, 224, interpolation="bilinear")(inputs)
    normalization = architecture.get("normalization", {})
    method = normalization.get("method")
    if method == "standard":
        mean = tf.constant(normalization["mean"], dtype=tf.float32)
        std = tf.constant(normalization["standard_deviation"], dtype=tf.float32)
        images = tf.keras.layers.Lambda(
            lambda value, scale=std, offset=mean: value * scale + offset,
            name="restore_image_scale",
        )(images)
    elif method == "none":
        if method == "none":
            images = tf.keras.layers.Rescaling(1.0 / 255.0)(images)
    elif method != "minmax":
        raise ValueError(f"Unsupported pretrained image normalization: {method}")
    if input_shape[-1] == 1:
        images = tf.keras.layers.Lambda(
            lambda value: tf.image.grayscale_to_rgb(value),
            name="grayscale_to_rgb",
        )(images)
    images = tf.keras.layers.Rescaling(1.0 / 255.0)(images)
    mean = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
    std = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
    images = tf.keras.layers.Lambda(
        lambda value: (value - mean) / std,
        name="imagenet_normalization",
    )(images)
    pixel_values = tf.keras.layers.Permute((3, 1, 2), name="channels_first")(images)
    hidden = backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
    logits = tf.keras.layers.Dense(num_classes, name="mnist_classifier")(hidden)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="pretrained_deit_tiny_mnist")
