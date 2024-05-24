import keras.utils
from keras.datasets import mnist
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

import matplotlib.pyplot as plt

import numpy as np
import os

import tensorflow as tf

tf = tf.compat.v1
sess = tf.Session()
K.set_session(sess)


class CGAN():
    def __init__(self):
        # Входные форматы
        self.IMG_SHAPE = (28, 28, 1)
        self.NUM_CLASSES = 10
        self.LATENT_DIM = 1

        """
        Генератор и Дискриминатор
        """
        # Мучаемся со входами
        self.image_inp = Input(shape=self.IMG_SHAPE, name="image")
        self.label_inp = Input(shape=(self.NUM_CLASSES,), name="label")
        self.latent_space_inp = Input(shape=(self.LATENT_DIM,), name="latent_space")

        # Создаем дискриминатор
        self.build_discriminator()
        self.discriminator.summary()
        # Создаем генератор
        self.build_generator()
        self.generator.summary()

        """
        Модели
        """
        # z == latten_space
        self.generated_z = self.generator([self.latent_space_inp, self.label_inp])
        self.dis_gen_z = self.discriminator([self.generated_z, self.label_inp])

        self.cgan_model = Model([self.latent_space_inp, self.label_inp], self.dis_gen_z, name="CGAN")
        self.cgan = self.cgan_model([self.latent_space_inp, self.label_inp])

        self.optimizer_gen = Adam(5e-4)  # У Генератора больше
        self.optimizer_dis = Adam(1e-5)  # У Дискриминатора меньше (чтоб не душил Генератор)

    def build_generator(self) -> Model:
        # Мучаемся со входом
        latent_space_and_label = concatenate([self.latent_space_inp, self.label_inp])

        # Сам Генератор
        x = Dense(5**2, activation=LeakyReLU(0.1))(latent_space_and_label)
        x = Reshape((5, 5, 1))(x)
        x = Conv2DTranspose(64, (3, 3), activation=LeakyReLU(0.1))(x)

        for i in range(5, 7):
            x = Conv2DTranspose(2**i, (3, 3), activation=LeakyReLU(0.1), padding="same")(x)
            x = UpSampling2D()(x)
            x = Conv2DTranspose(2**i, (3, 3), activation=LeakyReLU(0.1), padding="same")(x)

        x = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
        x = Reshape(self.IMG_SHAPE)(x)

        self.generator = Model([self.latent_space_inp, self.label_inp], x, name="generator")

    def build_discriminator(self) -> Model:
        # Объединяем картинку с лейблами
        repeat_n = int(np.prod(self.IMG_SHAPE))
        units_repeat = RepeatVector(repeat_n)(self.label_inp)
        units_repeat = Reshape([*self.IMG_SHAPE[:-1], self.NUM_CLASSES])(units_repeat)

        img_and_label = concatenate([units_repeat, self.image_inp])
        x = img_and_label

        # Сам Дискриминатор
        for i in range(8, 5, -1):
            x = Conv2D(2**i, (3, 3), activation=LeakyReLU(0.1), padding="same")(x)
            x = MaxPooling2D()(x)
            x = Conv2D(2**i, (3, 3), activation=LeakyReLU(0.1), padding="same")(x)

        x = Flatten()(x)
        x = Dense(1, activation="sigmoid")(x)

        self.discriminator = Model([self.image_inp, self.label_inp], x, name="discriminator")

    def get_batch(self, batch_size):
        # Загружаем набор данных
        (x, y), (x_, y_) = mnist.load_data()

        # Объединяем всё (куда добру пропадать)
        x = np.append(x, x_, axis=0)
        y = np.append(y, y_, axis=0)

        # Конфигурируем данные
        x = x.astype(np.float32) / 255.
        x = np.expand_dims(x, axis=3)

        y = y.reshape(-1, 1)
        y = keras.utils.to_categorical(y, self.NUM_CLASSES)

        n_batches = x.shape[0] // batch_size

        # Замыкание
        while True:
            # Перед игрой тасуем колоду
            idxs = np.random.permutation(y.shape[0])
            x = x[idxs]
            y = y[idxs]

            for i in range(n_batches - 1):
                batch_x = x[batch_size * i: batch_size * (i + 1)]
                batch_y = y[batch_size * i: batch_size * (i + 1)]
                noise = np.random.normal(0, 1, (batch_size, self.LATENT_DIM))
                yield batch_x, batch_y, noise

    def train(self, epochs, batch_size=128, sample_interval=100):
        # Просто единицы и нули для Дискриминатора
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        get_batch = self.get_batch(batch_size)

        for iter_learn in range(epochs):
            # -------------------------
            #  Обучение дискриминатора
            # -------------------------
            images, labels, noise = next(get_batch)
            with tf.GradientTape() as dis_tape:
                dis_real_output = self.discriminator([images, labels], training=True)
                generated_images = self.generator([noise, labels], training=False)
                dis_fake_output = self.discriminator([generated_images, labels], training=True)

                # Чем настоящие картинки нереальнее и сгенерированные реальные, тем ошибка больше
                l_dis = 0.5 * (tf.reduce_mean(-tf.math.log(dis_real_output + 1e-8)) +
                               tf.reduce_mean(-tf.math.log(1. - dis_fake_output + 1e-8)))

            # Получаем градиенты для дискриминатора
            grads_dis = dis_tape.gradient(l_dis, self.discriminator.trainable_variables)

            # Применяем градиенты
            self.optimizer_dis.apply_gradients(zip(grads_dis, self.discriminator.trainable_variables))

            # ---------------------
            #  Обучение генератора
            # ---------------------
            images, labels, noise = next(get_batch)
            with tf.GradientTape() as gen_tape:
                generated_images = self.generator([noise, labels], training=True)
                dis_output = self.discriminator([generated_images, labels], training=False)

                # Чем более реалистичная картина (для дискриминатора), тем меньше ошибка
                l_gen = -tf.reduce_mean(tf.math.log(dis_output + 1e-8))

            # Получаем градиенты для генератора
            grads_gen = gen_tape.gradient(l_gen, self.generator.trainable_variables)

            # Применяем градиенты
            self.optimizer_gen.apply_gradients(zip(grads_gen, self.generator.trainable_variables))

            # ______________________________
            # Сохраняем генерируемые образцы
            if iter_learn % sample_interval == 0:
                self.sample_images(iter_learn)

                # Вывод прогресса
                print(f"{iter_learn:03} \t"
                      f"[Dis loss: {l_dis:.3f}] \t"
                      f"[Gen loss: {l_gen:.3f}]")

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.LATENT_DIM))
        label = np.arange(0, self.NUM_CLASSES).reshape(-1, 1)
        sampled_labels = keras.utils.to_categorical(label, self.NUM_CLASSES)

        gen_imgs = self.generator.predict([noise, sampled_labels], verbose=False)

        # Делаем картинку
        fig, axs = plt.subplots(r, c, figsize=(13, 6))  # Увеличиваем размер фигуры
        count = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap="gray")
                axs[i, j].set_title(label[count][0])
                axs[i, j].axis("off")
                axs[i, j].set_aspect("equal")  # Сохраняем пропорции картинки
                count += 1
        fig.savefig("mnist_images/%d.png" % epoch)
        plt.close()


if __name__ == "__main__":
    # Удаляем все прошлые изображения
    for i in os.listdir("./mnist_images"):
        os.remove(f"./mnist_images/{i}")

    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=333, sample_interval=200)
