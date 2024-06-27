from tensorflow import keras
import tensorflow as tf

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, RMSprop
import keras.backend as K

from math import log2
import numpy as np
from random import randint

import matplotlib.pyplot as plt

img_side = 252


# Разбиваем датасет на тренировочную группу и группу валидации
def init_data_with_batch_size(batch_size, dataset="flowers_dataset"):
    """Чтобы использовать "new_flowers" (расширенный датасет) надо запустить increasing_data.py"""
    train_data = keras.preprocessing.image_dataset_from_directory(
        dataset,
        image_size=(img_side, img_side),
        label_mode="categorical",
        shuffle=True,
        batch_size=batch_size,
    )

    # Добавляем лейблы (т.к. у нас Cvcae)
    return train_data.map(lambda x, y: (x/255., y))


# Константы
filters = 80   # Верхняя граница
dropout = 0.0
hidden_units = 32
hidden_img_shape = [32, 32]
core_size = (3, 3)

# Чем меньше тем лучше:
amount_encode_layers = 4
amount_decode_layers = 3


class Sampling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


"""Энкодер"""
encoder_input = Input(shape=(img_side, img_side, 3), name="encoder_input")
x = encoder_input

for i in range(amount_encode_layers):
    x = MaxPool2D()(x)
    x = Dropout(dropout)(x)
    x = Conv2D(filters // 2**i, core_size, activation=LeakyReLU())(x)
    x = Conv2D(filters // 2**i, core_size, activation=LeakyReLU())(x)

x = Flatten()(x)
label_input = Input(shape=(5,), name="label_input")

# Постепенно сжимаем до размера скрытого вектора
dense_units = x.shape[-1]
while dense_units >= hidden_units:
    x = Dropout(dropout)(x)
    # Добавляем метки класса
    x = concatenate([label_input, x])
    x = Dense(dense_units, activation=LeakyReLU())(x)
    dense_units //= 2

# Добавляем метки класса
x = concatenate([label_input, x])

# Сездаём слои для контронтроля скрытого пространства
z_mean = Dense(hidden_units, name="z_mean")(x)
z_log_var = Dense(hidden_units, name="z_log_var")(x)
z = Sampling(name="z")([z_mean, z_log_var])

encoder = Model([encoder_input, label_input], [z_mean, z_log_var, z], name="encoder")


"""Декодер"""
decoder_input = keras.Input(shape=(hidden_units,), name="decoder_input")

# Добавляем метки класса
label_input = Input(shape=(5,), name="label_input")
x = concatenate([decoder_input, label_input])

# Постепенно разжимаем от размера скрытого вектора до hidden_img_shape
dense_units = hidden_units * 2
while dense_units * 2 <= hidden_img_shape[0] * hidden_img_shape[1]:
    x = Dropout(dropout)(x)
    # Добавляем метки класса
    x = concatenate([label_input, x])
    x = Dense(dense_units, activation=LeakyReLU())(x)
    dense_units *= 2

# Снова добавляем метки класса
x = concatenate([label_input, x])

# Разжимаем вектор признаков в маленькую картинку
x = Dense(hidden_img_shape[0] * hidden_img_shape[1], activation=LeakyReLU())(x)
x = Reshape(hidden_img_shape + [1])(x)

# Расширяем карту признаков, увеличиваем картинку и количество фильтров
for i in range(amount_decode_layers-1, -1, -1):
    x = Dropout(dropout)(x)
    x = UpSampling2D()(x)
    # x = Conv2D(filters // 2**i, core_size, activation=LeakyReLU())(x)
    x = Conv2D(filters // 2**i, core_size, activation=LeakyReLU())(x)

# Увеличиваем чёткость при помощи Dense (берём строку картинки)
# (проходимся по каждому каналу отдельно)
x_temp = x
x = Reshape((filters * x_temp.shape[1], x_temp.shape[1]))(x_temp)
x = Dropout(dropout)(x)
x = Dense(x_temp.shape[1], activation="sigmoid")(x)
x = Reshape((x_temp.shape[1], x_temp.shape[1], filters))(x)
x = add([x_temp, x])

decoded_img = Conv2D(3, core_size, activation="sigmoid")(x)
decoder = Model([decoder_input, label_input], decoded_img, name="decoder")


"""CVCAE"""


class CVCAE(keras.Model):
    """Я без понятия как это работает"""
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.decode_loss_tracker = keras.metrics.Mean(
            name="decode_loss"
        )
        self.bias_loss_tracker = keras.metrics.Mean(name="bias_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.decode_loss_tracker,
            self.bias_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            decoded = self.decoder([z, data[1]])
            decode_loss = K.mean(
                    keras.losses.binary_crossentropy(data[0], decoded),
            )
            bias_loss = -0.5 * (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
            bias_loss = K.mean(bias_loss, axis=1)

            decode_loss *= 1000
            bias_loss /= 2

            total_loss = decode_loss + bias_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Обновляем метрики и возвращаем результат
        self.total_loss_tracker.update_state(total_loss)
        self.decode_loss_tracker.update_state(decode_loss)
        self.bias_loss_tracker.update_state(bias_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "decode_loss": self.decode_loss_tracker.result(),
            "bias_loss": self.bias_loss_tracker.result(),
        }


# Выводим количество параметров
encoder.summary()
decoder.summary()
print("Encoder:", f"{encoder.count_params():,}")
print("Decoder:", f"{decoder.count_params():,}")
print("Sum:    ", f"{decoder.count_params() + encoder.count_params():,}")

cvcae = CVCAE(encoder, decoder, name="CVCAE")
cvcae.compile(optimizer=Adam(1e-3))

data = init_data_with_batch_size(12, "flowers_dataset")
cvcae.fit(
    data,
    epochs=100,
)


def show_row_images(raw_data):
    # Ограничиваемся только 32 восстановленными изображениями (чтобы считать меньше)
    data = np.array([i[0][0] for count, i in enumerate(raw_data) if count < 22])
    labels = np.array([i[1][0] for count, i in enumerate(raw_data) if count < 22])
    _, _, encoded = encoder.predict([data, labels], verbose=False)
    generated_images = decoder.predict([encoded, labels], verbose=False)

    num_images = 4

    plt.figure(figsize=(20, 11))

    for _ in range(num_images):
        random_num = randint(0, 22-1)

        # Оригинальное изображение
        plt.subplot(2, num_images, _ + 1)
        plt.imshow(data[random_num])
        plt.gray()
        plt.title("Train")
        plt.axis("off")

        # Сгенерированное изображение
        plt.subplot(2, num_images, _ + num_images + 1)
        plt.imshow(generated_images[random_num])
        plt.gray()
        plt.title("Generated")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


raw_data = init_data_with_batch_size(1, "flowers_dataset")
for _ in range(3):
    show_row_images(raw_data)


d = init_data_with_batch_size(1, "flowers_dataset")
data = np.array([next(iter(d))[0][0] for _ in range(1000)])
labels = np.array([next(iter(d))[1][0] for _ in range(1000)])

_, _, all_hidden_data = encoder.predict([data, labels])
mean, std = np.mean(all_hidden_data), np.std(all_hidden_data)
print("mean (ideal: 0):", mean)
print("std  (ideal: 1):", std)

for _ in range(4):
    num_images = 4

    noise = np.random.normal(mean, std, [num_images*2, hidden_units])
    label = np.array([keras.utils.to_categorical(np.random.randint(0, 5), 5)
                      for _ in range(num_images*2)])
    generated_images = np.array(decoder.predict([noise, label], verbose=False))

    plt.figure(figsize=(20, 11))

    for i in range(num_images):
        # Оригинальное изображение
        plt.subplot(2, num_images, i + 1)

        # Переводим в промежуток [0; 1]
        plt.imshow(generated_images[i + num_images])
        plt.axis("off")

        # Сгенерированное изображение
        plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(generated_images[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


"""Выводим Архитектуру"""
encoder_img = tf.keras.utils.plot_model(encoder, to_file="./images/encoder.png", show_shapes=False, show_layer_names=False,
                                        dpi=128, show_layer_activations=False)

decoder_img = tf.keras.utils.plot_model(decoder, to_file="./images/decoder.png", show_shapes=False, show_layer_names=False,
                                        dpi=128, show_layer_activations=False)

cvcae.save_weights("cvcae")
# cvcae.load_weights("cvcae")
