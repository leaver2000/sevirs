from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf

from .._typing import Tensor

if TYPE_CHECKING:
    from keras.api._v2.keras import Model, layers
else:
    from tensorflow.keras import Model, layers  # type: ignore


class CasualSelfAttention(Model):
    def __init__(self, d_model: int, num_heads: int, rate: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rate = rate

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
            q (Tensor): query shape == (..., seq_len_q, depth)
            k (Tensor): key shape == (..., seq_len_k, depth)
            v (Tensor): value shape == (..., seq_len_v, depth_v)
            mask (Tensor, optional): Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
            tuple[Tensor, Tensor]: output, attention_weights
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # add the mask to the
        if mask is not None:
            scaled_attention_logits += mask * -1e9
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        return output, attention_weights


class MultiLayerPerceptron(Model):
    def __init__(self, d_model: int, dff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.dense1 = layers.Dense(dff, activation="relu")
        self.dense2 = layers.Dense(d_model)

    def call(self, x: Tensor) -> Tensor:
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class LayerNormilization(Model):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x: Tensor) -> Tensor:
        return self.layer(x)


class Block(Model):
    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = CasualSelfAttention(d_model, num_heads, rate)
        self.mlp = MultiLayerPerceptron(d_model, dff)

        self.layernorm1 = LayerNormilization(d_model)
        self.layernorm2 = LayerNormilization(d_model)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x: Tensor, training: bool, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        mlp_output = self.mlp(out1)  # (batch_size, input_seq_len, d_model)
        mlp_output = self.dropout2(mlp_output, training=training)
        out2 = self.layernorm2(out1 + mlp_output)  # (batch_size, input_seq_len, d_model)
        return out2, attn_output


class VisionTransformer(Model):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_shape: tuple[int, int, int],
        num_classes: int,
        rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.rate = rate

        self.pos_encoding = self.positional_encoding()
        self.input_layer = layers.Input(shape=input_shape)
        self.flatten = layers.Flatten()
        self.embedding = layers.Dense(d_model)
        self.dropout = layers.Dropout(rate)
        self.blocks = [Block(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.classifier = layers.Dense(num_classes)

    def call(self, x: Tensor, training: bool) -> Tensor:
        seq_len = tf.shape(x)[1]
        x = self.flatten(x)
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, _ = self.blocks[i](x, training)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def positional_encoding(self) -> Tensor:
        """PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        """
        pos = tf.range(self.input_shape[0], dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / self.d_model)
        angle_rads = pos * angle_rates
        # apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def to_disk(self, path: str) -> None:
        self.save_weights(path)

    @classmethod
    def from_disk(
        cls,
        path: str,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_shape: tuple[int, int, int],
        num_classes: int,
        rate: float = 0.1,
    ) -> VisionTransformer:
        model = cls(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_shape,
            num_classes,
            rate,
        )
        model.load_weights(path)
        return model
