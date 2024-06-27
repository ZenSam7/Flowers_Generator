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
import os

img_shape = 190
batch_size = 1
epoches = 1024
noise_units = 8
hidden_img = (3, 3, 1)

amount_layers = 6
filters = 16  # Нижняя граница


train_data = keras.preprocessing.image_dataset_from_directory(
    "flowers_dataset",
    image_size=(img_shape, img_shape),
    label_mode="categorical",
    shuffle=True,
    batch_size=batch_size,
)


def batch_generator():
    """Чтобы использовать "big_flowers_dataset" (расширенный датасет) надо запустить increasing_data.py"""
    for _ in range(epoches):
        # Добавляем лейблы и нормализуем в [-1; 1]
        x, y = next(iter(train_data))
        x = x / 255. * 2 - 1
        noise = np.random.normal(0, 1, (batch_size, noise_units))
        yield [noise, y], x


flowers_path = "./generated_flowers"
# Удаляем все прошлые изображения
for i in os.listdir(flowers_path):
    os.remove(f"{flowers_path}/{i}")


"""Просто строим генератор"""
generator_input = keras.Input(shape=(noise_units,), name="decoder_input")

# Добавляем метки класса
label_input = Input(shape=(5,), name="label_input")
x = concatenate([generator_input, label_input])

# Разжимаем вектор признаков в маленькую картинку
x = Dense(np.prod(hidden_img), activation=LeakyReLU())(x)
x = Reshape(hidden_img)(x)

# Расширяем карту признаков, увеличиваем картинку и количество фильтров
for i in range(amount_layers - 1, -1, -1):
    # x = Dropout(0.2)(BatchNormalization()(x))
    x = Conv2DTranspose(filters * 2**i, (4, 4), strides=2, padding="same", activation=LeakyReLU())(x)

generated_img = Conv2D(3, (3, 3), activation="tanh")(x)
generator = Model([generator_input, label_input], generated_img, name="decoder")

# Выводим количество параметров
generator.summary()
generator.compile(optimizer=Adam(1e-3), loss="mae")


"""Выводим изображения каждые ... эпох"""
for epoch in range(10**10):
    # Обучаем
    generator.fit(batch_generator(), epochs=1)

    # Выводим
    row, column = 2, 5
    noise = np.random.normal(0, 1, (row * column, noise_units))
    label = np.array([
        np.arange(0, 5) for _ in range(row)
    ]).reshape((-1, 1))
    sampled_labels = keras.utils.to_categorical(label, 5)

    gen_imgs = generator.predict([noise, sampled_labels], verbose=False)
    gen_imgs = np.array(gen_imgs)
    gen_imgs -= np.min(gen_imgs)
    gen_imgs /= np.max(gen_imgs)

    # Делаем картинку
    fig, axs = plt.subplots(row, column, figsize=(12, 6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

    count = 0
    for i in range(row):
        for j in range(column):
            axs[i, j].imshow(gen_imgs[count, :, :, :])
            axs[i, j].set_title(label[count][0])
            axs[i, j].axis("off")
            count += 1

    fig.savefig(f"{flowers_path}/%d.png" % epoch, dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close()
