from __future__ import print_function, division

import keras.utils
from keras.datasets import mnist
from keras.layers import *
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
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
        self.LATENT_DIM = 8

        """
        Генератор и Дискриминатор
        """
        # Если Дискриминатор обыгрывает Генератор, то обучение остановится
        # Поэтому в train() сть внутренние циклы, чтобы дать фору
        self.NUM_STEP_LEARN = 5

        # Мучаемся со входами
        # x_ = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="image")
        # y_ = tf.placeholder(tf.float32, shape=(None, self.NUM_CLASSES), name="label")
        # z_ = tf.placeholder(tf.float32, shape=(None, self.LATENT_DIM), name="latent_space")

        self.image_inp = Input(shape=self.IMG_SHAPE, name="image")
        self.label_inp = Input(shape=(self.NUM_CLASSES,), name="label")
        self.latent_space_inp = Input(shape=(self.LATENT_DIM,), name="latent_space")

        # Создаем дискриминатор
        self.build_discriminator()
        self.discriminator.compile(
            loss=["binary_crossentropy"],
            optimizer=Adam(1e-3),
            metrics=["accuracy"],
        )
        self.discriminator.summary()

        # Создаем генератор
        self.build_generator()
        self.generator.summary()

        """
        Модели
        """
        # z == latten_space
        self.generated_z = self.generator([self.latent_space_inp, self.label_inp])

        self.dis_img = self.discriminator([self.image_inp, self.label_inp])
        self.dis_gen_z = self.discriminator([self.generated_z, self.label_inp])

        self.cgan_model = Model([self.latent_space_inp, self.label_inp], self.dis_gen_z, name="CGAN")
        self.cgan = self.cgan_model([self.latent_space_inp, self.label_inp])

        """
        Ошибки
        """
        self.optimizer_gen = Adam(5e-4)  # У Генератора больше
        self.optimizer_dis = Adam(1e-4)  # У Дискриминатора меньше (чтоб не душил Генератор)

        # # Переменные генератора и дискриминаторы (отдельно) для оптимизаторов
        # self.generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
        # self.discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

        # Получаем переменные генератора и дискриминатора
        self.generator_vars = self.generator.trainable_variables
        self.discriminator_vars = self.discriminator.trainable_variables

        # Инициализируем
        # sess.run(tf.global_variables_initializer())

    def step_learn_gen(self, image, label, latent):
        """Шаг обучения Генератора"""
        with tf.GradientTape() as gen_tape:
            # Передаем данные через генератор
            generated_images = self.generator([latent, label], training=True)
            # Передаем сгенерированные изображения через дискриминатор
            dis_output = self.discriminator([generated_images, label], training=False)
            # Вычисляем потери генератора
            l_gen = -tf.reduce_mean(tf.math.log(1. - dis_output + 1e-9))

        # Получаем градиенты для генератора
        grads_gen = gen_tape.gradient(l_gen, self.generator.trainable_variables)

        # Применяем градиенты
        self.optimizer_gen.apply_gradients(zip(grads_gen, self.generator.trainable_variables))

        return l_gen  # Возвращаем значение потерь

    def step_learn_dis(self, image, label, latent):
        """Шаг обучения Дискриминатора"""
        with tf.GradientTape() as dis_tape:
            # Передаем реальные изображения через дискриминатор
            dis_real_output = self.discriminator([image, label], training=True)
            # Генерируем изображения
            generated_images = self.generator([latent, label], training=False)
            # Передаем сгенерированные изображения через дискриминатор
            dis_fake_output = self.discriminator([generated_images, label], training=True)
            # Вычисляем потери дискриминатора
            l_dis = 0.5 * (tf.reduce_mean(-tf.math.log(dis_real_output + 1e-9)) +
                           tf.reduce_mean(-tf.math.log(1. - dis_fake_output + 1e-9)))

        # Получаем градиенты для дискриминатора
        grads_dis = dis_tape.gradient(l_dis, self.discriminator.trainable_variables)

        # Применяем градиенты
        self.optimizer_dis.apply_gradients(zip(grads_dis, self.discriminator.trainable_variables))

        return l_dis  # Возвращаем значение потерь

    def build_generator(self) -> Model:
        # Мучаемся со входом
        with tf.variable_scope("generator"):
            latent_space_and_label = concatenate([self.latent_space_inp, self.label_inp])

            # Сам Генератор
            x = Dense(7**2*64, activation=LeakyReLU())(latent_space_and_label)
            x = Reshape((7, 7, 64))(x)

            for i in [64, 16]:
                x = Dropout(0.1)(x)
                x = Conv2D(i, (3, 3), activation=LeakyReLU(), padding="same")(x)
                x = Conv2D(i, (3, 3), activation=LeakyReLU(), padding="same")(x)
                x = UpSampling2D()(x)

            x = Conv2D(1, (7, 7), activation="sigmoid", padding="same")(x)
            x = Reshape(self.IMG_SHAPE)(x)

            self.generator = Model([self.latent_space_inp, self.label_inp], x, name="generator")

    def build_discriminator(self) -> Model:
        with tf.variable_scope("discriminator"):
            # Объединяем картинку с лейблами
            repeat_n = int(np.prod(self.IMG_SHAPE))
            units_repeat = RepeatVector(repeat_n)(self.label_inp)
            units_repeat = Reshape([*self.IMG_SHAPE[:-1], self.NUM_CLASSES])(units_repeat)

            img_and_label = concatenate([units_repeat, self.image_inp])

            # Сам Дискриминатор
            x = img_and_label
            for i in range(2, 5):
                x = Dropout(0.1)(x)
                x = Conv2D(2**i, (3, 3), activation=LeakyReLU(), padding="same")(x)
                x = MaxPooling2D()(x)

            x = Flatten()(x)
            x = Dense(1, activation="sigmoid")(x)

            self.discriminator = Model([self.image_inp, self.label_inp], x, name="discriminator")

    def batch_generator(self, batch_size):
        """Декоратор для генератора батчей"""
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
        global loss_dis, loss_gen
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        get_batch = self.batch_generator(batch_size)

        for iter_learn in range(epochs):
            # -------------------------
            #  Обучение дискриминатора
            # -------------------------
            for _ in range(self.NUM_STEP_LEARN):
                imgs, labels, noise = next(get_batch)
                loss_dis = self.step_learn_dis(imgs, labels, noise)

                # Если Дискриминатор обыгрывает Генератор, то обучение остановится
                if loss_dis < 1.0:
                    break

            # ---------------------
            #  Обучение генератора
            # ---------------------
            for _ in range(self.NUM_STEP_LEARN):
                imgs, labels, noise = next(get_batch)
                loss_gen = self.step_learn_gen(imgs, labels, noise)

                # Если Генератор сильно обыгрывает, то обучение остановится
                if loss_gen > 0.5:
                    break

            # Сохраняем генерируемые образцы
            if iter_learn % sample_interval == 0:
                self.sample_images(iter_learn)

                # Вывод прогресса
                print(f"{iter_learn:03} \t"
                      f"[Dis loss: {loss_dis:.3f}] \t"
                      f"[Gen loss: {loss_gen:.3f}]")

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.LATENT_DIM))
        label = np.arange(0, 10).reshape(-1, 1)
        sampled_labels = keras.utils.to_categorical(label, self.NUM_CLASSES)

        gen_imgs = self.generator.predict([noise, sampled_labels], verbose=False)

        # Переводим в промежуток [0; 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

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
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == "__main__":
    # Удаляем все прошлые изображения
    for i in os.listdir("./images"):
        os.remove(f"./images/{i}")

    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=512, sample_interval=100)
