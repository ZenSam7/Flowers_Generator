"""
https://keras.io/examples/generative/vae/
"""
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from keras import layers
import tensorflow as tf
from tensorflow import keras
import keras.backend as K


## Create a sampling layer

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


## Build the encoder

# У нас все 10 цифр, а в latend_dim мы передаём только стиль написания цифры
latent_dim = 1

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation="relu")(x)

# Include input layer for class labels (всего 10 классов)
label_inputs = keras.Input(shape=(10,))
x = layers.concatenate([x, label_inputs])

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model([encoder_inputs, label_inputs], [z_mean, z_log_var, z], name="encoder")
encoder.summary()


## Build the decoder

latent_inputs = keras.Input(shape=(latent_dim,))
# Include input layer for class labels (всего 10 классов)
label_inputs = keras.Input(shape=(10,))
x = layers.concatenate([latent_inputs, label_inputs])

x = layers.Dense(7 * 7 * 32, activation="relu")(x)
x = layers.Reshape((7, 7, 32))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model([latent_inputs, label_inputs], decoder_outputs, name="decoder")
decoder.summary()


## Define the VAE as a `Model` with a custom `train_step`

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        data = data[0]  # data == ( (img, labels) )
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder([z, data[1]])  # Передаём метки
            reconstruction_loss = K.mean(
                K.sum(
                    keras.losses.binary_crossentropy(data[0], reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
            kl_loss = K.mean(K.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


## Train the VAE

(x_train, labels_train), (x_test, labels_test) = keras.datasets.mnist.load_data()

# Объединяем тренировочную и тестовую выборку
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_labels = np.concatenate([labels_train, labels_test], axis=0)
# Обрабатываем
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
mnist_labels = keras.utils.to_categorical(mnist_labels, 10)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit([mnist_digits, mnist_labels], epochs=25, batch_size=1024)


## Display a grid of sampled digits

import matplotlib.pyplot as plt


def plot_latent_space(vae, figsize=15):
    # У нас все 10 цифр, а в latend_dim мы передаём только стиль написания цифры
    n = 10

    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n))

    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_y = np.linspace(scale, -scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j in range(10):
            z_sample = np.array([yi])
            label_sample = keras.utils.to_categorical(j, 10)
            label_sample = label_sample.reshape(1, -1)

            x_decoded = vae.decoder.predict([z_sample, label_sample], verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_y = np.round(grid_y, 1)
    plt.yticks(pixel_range, sample_range_y)
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent_space(vae)

