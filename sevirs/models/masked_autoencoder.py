"""Masked Autoencoding of geospatial data for pretraining."""
from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from keras.api._v2.keras import Model, layers
    from keras.src.engine.functional import Functional as F  # noqa

else:
    from tensorflow.keras import Model, layers  # type: ignore


class Autoencoder(Model):
    """
    tensorflow generative [autoencoder](https://www.tensorflow.org/tutorials/generative/autoencoder)
    """

    def __init__(self, latent_dim: int, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [layers.Dense(tf.math.reduce_prod(shape), activation="sigmoid"), layers.Reshape(shape)]
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        encoded = self.encoder.__call__(x)
        decoded = self.decoder.__call__(encoded)
        return decoded


def main():
    import numpy as np

    N_SAMPLES = 2
    C = 3  # channels (radar products, ...)
    L = 64  # latitude
    W = 64  # longitude
    T = 49  # time steps
    arr = np.random.rand(N_SAMPLES, C, T, L, W)

    shape = arr.shape[1:]
    latent_dim = 64
    autoencoder = Autoencoder(latent_dim, shape)
    autoencoder.compile(optimizer="adam", loss="mse")
    # autoencoder.fit(arr, arr, epochs=10, shuffle=True, validation_split=0.1)
    save_path = "autoencoder.h5"
    autoencoder.save(save_path)
    autoencoder = tf.keras.models.load_model(save_path)


if __name__ == "__main__":
    main()
