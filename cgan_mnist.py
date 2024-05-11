from __future__ import print_function, division

import keras.utils
from keras.datasets import mnist
from keras.layers import (Input, Dense, Reshape, Flatten, Dropout, multiply, MaxPooling2D,
                          BatchNormalization, Activation, Embedding, ZeroPadding2D, LeakyReLU, concatenate)
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np


class CGAN():
    def __init__(self):
        # Входные форматы
        self.img_shape = (28, 28, 1)
        self.num_classes = 10
        self.latent_dim = 16

        # Создаем дискриминатор
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy'],
            optimizer=Adam(1e-4),
            metrics=['accuracy'],
        )
        self.discriminator.summary()

        # Создаем генератор
        self.generator = self.build_generator()
        self.discriminator.compile(
            loss=['binary_crossentropy'],
            optimizer=Adam(1e-3),
            metrics=['accuracy'],
        )
        self.generator.summary()

        # Объединяем их
        noise = Input(shape=(self.latent_dim,), name="noise_combine")
        label = Input(shape=(self.num_classes,), name="label_combine")
        img = self.generator([noise, label])

        # В combined мы обучем только Генератор, т.к. чтобы обучить Генератор надо
        # чтобы градиент прошёл через Дискриминатор (Дискриминатор обучаем отдельно)
        self.discriminator.trainable = False

        valid = self.discriminator([img, label])

        # Комбинированная модель (генератор и дискриминатор)
        # Обучает генератор, чтобы обмануть дискриминатор
        self.combined = Model([noise, label], valid)
        self.combined.compile(
            loss=['binary_crossentropy'],
            optimizer=Adam(1e-3),
        )

    def build_generator(self) -> Model:
        # Мучаемся со входом
        latent_space = Input(shape=(self.latent_dim,), name="latent_space")

        label = Input(shape=(self.num_classes,), name="label")

        latent_space_and_label = concatenate([latent_space, label])

        # Сам Генератор
        x = Dense(7 * 7, activation=LeakyReLU())(latent_space_and_label)
        x = Reshape((7, 7, 1))(x)

        x = Conv2D(4, (3, 3), activation=LeakyReLU(), padding="same")(x)
        x = Conv2D(8, (3, 3), activation=LeakyReLU(), padding="same")(x)
        x = UpSampling2D()(x)

        x = Conv2D(16, (5, 5), activation=LeakyReLU(), padding="same")(x)
        x = Conv2D(32, (5, 5), activation=LeakyReLU(), padding="same")(x)
        x = UpSampling2D()(x)

        x = Conv2D(16, (7, 7), activation=LeakyReLU(), padding="same")(x)
        x = Conv2D(32, (7, 7), activation=LeakyReLU(), padding="same")(x)

        x = Conv2D(1, (9, 9), activation="tanh", padding="same")(x)
        output_img = Reshape(self.img_shape)(x)

        return Model([latent_space, label], output_img)

    def build_discriminator(self) -> Model:
        # Использауем Эмбеддинг
        img = Input(self.img_shape, name="img")
        img_flatten = Flatten()(img)

        label = Input(shape=(self.num_classes,), name="label")

        img_and_label = concatenate([label, img_flatten])
        x = Dense(np.prod(self.img_shape))(img_and_label)
        x = Reshape(self.img_shape)(x)

        for i in range(4, 0, -1):
            x = Dropout(0.1)(x)
            x = Conv2D(2**i, (2*i+1, 2*i+1), activation=LeakyReLU(), padding="same")(x)
            x = Conv2D(2**i, (2*i+1, 2*i+1), activation=LeakyReLU(), padding="same")(x)
            x = MaxPooling2D()(x)

        x = Flatten()(x)
        validity = Dense(1, activation="sigmoid")(x)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Загружаем набор данных
        (X_train, y_train), (_, _) = mnist.load_data()

        # Конфигурируем входные данные
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        y_train = y_train.reshape(-1, 1)
        y_train = keras.utils.to_categorical(y_train, self.num_classes)

        # Просто единицы и нули для Дискриминатора
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Обучение дискриминатора
            # ---------------------

            # Выбираем случайный пакет картинок
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Генерируем пакет новых картинок
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict([noise, labels], verbose=False)

            # Обучаем дискриминатор распознавать настоящие и сгенерированные изображения
            dis_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            dis_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            dis_loss = 0.5 * np.add(dis_loss_real, dis_loss_fake)

            # ---------------------
            #  Обучение генератора
            # ---------------------

            # Случайная метка
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            sampled_labels = keras.utils.to_categorical(sampled_labels, self.num_classes)

            # Обучаем генератор (но не дискриминатор)
            gen_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Сохраняем генерируемые образцы
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

                # Вывод прогресса
                print(f"{epoch} \t"
                      f"[D loss: {dis_loss[0]:.3f}, acc.: {100*dis_loss[1]:.0f}]\t"
                      f"[G loss: {gen_loss:.3f}]")

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        label = np.arange(0, 10).reshape(-1, 1)
        sampled_labels = keras.utils.to_categorical(label, self.num_classes)

        gen_imgs = self.generator.predict([noise, sampled_labels], verbose=False)

        # Переводим в промежуток [0; 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        # Делаем картинку
        fig, axs = plt.subplots(r, c)
        count = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap="gray")
                axs[i, j].set_title(label[count][0])
                axs[i, j].axis("off")
                count += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=256, sample_interval=500)
