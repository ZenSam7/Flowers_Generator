from tensorflow import keras
import tensorflow as tf

from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
import keras.backend as K

from math import log2
import numpy as np
from random import randint

import matplotlib.pyplot as plt
import os


# Удаляем все прошлые изображения
for i in os.listdir("./generated_flowers"):
    os.remove(f"./generated_flowers/{i}")


class CCGAN(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.NUM_CLASSES = 5  # Не менять!

        # Входные форматы
        self.IMG_SHAPE = (170, 170, 3)
        self.LATENT_DIM = 8
        self.HANDICAP = 5  # Фора чтобы одна иишка не отставала от другой

        # Константы
        self.FILTERS_DIS = 16  # Нижняя граница
        self.FILTERS_GEN = 80
        self.DROPOUT = 0.2
        self.HIDDEN_IMG_SHAPE = (80, 80, 1)

        # Чем меньше тем лучше:
        self.AMOUNT_DISCRIMINATOR_LAYERS = 3
        self.AMOUNT_GENERATOR_LAYERS = 3

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

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.decode_loss_tracker = keras.metrics.Mean(name="decode_loss")
        self.bias_loss_tracker = keras.metrics.Mean(name="bias_loss")

        """
        Модели
        """
        # z == latten_space (зображение между нейронками, которое мы выводим)
        self.generated_z = self.generator([self.latent_space_inp, self.label_inp])
        self.dis_gen_z = self.discriminator([self.generated_z, self.label_inp])

        self.ccgan = Model([self.latent_space_inp, self.label_inp], self.dis_gen_z, name="CCGAN")

        self.ccgan.compile(Adam(1e-3), loss="binary_crossentropy")
        self.generator.compile(Adam(1e-3), loss="binary_crossentropy")
        self.discriminator.compile(Adam(1e-3), loss="binary_crossentropy")

        self.optimizer_gen = Adam(1e-3)
        self.optimizer_dis = Adam(1e-3)

    def build_discriminator(self) -> Model:
        # Объединяем картинку с лейблами
        # x = Embedding(self.NUM_CLASSES, self.IMG_SHAPE[0]**2)(self.label_inp)
        # x = Reshape([*self.IMG_SHAPE[:2], self.NUM_CLASSES])(x)
        # x = concatenate([BatchNormalization()(self.image_inp), x])
        x = BatchNormalization()(self.image_inp)

        for i in range(1, self.AMOUNT_DISCRIMINATOR_LAYERS + 1):
            x = MaxPooling2D()(x)
            x = Dropout(self.DROPOUT)(BatchNormalization()(x))
            for _ in range(5):
                x = Conv2D(self.FILTERS_DIS * 2**i, (3, 3), activation=LeakyReLU())(x)

        # Максимально сжимаем изображение
        while x.shape[1] >= 3:
            x = Dropout(self.DROPOUT)(BatchNormalization()(x))
            x = Conv2D(x.shape[-1], (3, 3), activation=LeakyReLU())(x)

        x = Flatten()(x)

        # Постепенно сжимаем
        dense_units = x.shape[-1] // 2
        while dense_units >= 4:
            x = Dropout(self.DROPOUT)(BatchNormalization()(x))
            x = concatenate([self.label_inp, x])  # Добавляем метки класса
            x = Dense(dense_units, activation=LeakyReLU())(x)
            dense_units //= 2

        # Добавляем метки класса
        x = concatenate([self.label_inp, x])
        x = Dense(1, activation="sigmoid")(x)

        self.discriminator = Model([self.image_inp, self.label_inp], x, name="discriminator")

    def build_generator(self) -> Model:
        # Разжимаем вектор шума в маленькую картинку
        x = self.latent_space_inp

        for _ in range(self.AMOUNT_GENERATOR_LAYERS):
            x = Dropout(self.DROPOUT)(BatchNormalization()(x))
            x = concatenate([x, self.label_inp])
            x = Dense(x.shape[-1]*2, activation=LeakyReLU())(x)

        x = concatenate([x, self.label_inp])
        x = Dense(np.prod(self.HIDDEN_IMG_SHAPE), activation=LeakyReLU())(x)
        x = Reshape(self.HIDDEN_IMG_SHAPE)(x)

        x = BatchNormalization()(x)
        x = UpSampling2D()(x)
        for _ in range(6):
            x = Conv2DTranspose(self.FILTERS_GEN, (3, 3), activation=LeakyReLU())(x)

        x = BatchNormalization()(x)
        generated_img = Conv2D(3, (3, 3), activation="tanh")(x)

        self.generator = Model([self.latent_space_inp, self.label_inp], generated_img, name="generator")

    def batch_gen(self, batch_size, dataset):
        """Чтобы использовать "big_flowers_dataset" (расширенный датасет) надо запустить increasing_data.py"""
        train_data = keras.preprocessing.image_dataset_from_directory(
            dataset,
            image_size=self.IMG_SHAPE[:-1],
            label_mode="categorical",
            shuffle=True,
            batch_size=batch_size,
        )

        while True:
            # Добавляем лейблы (т.к. у нас CCGAN) и нормализуем в [-1; 1], т.к. юзаем tanh
            # (т.к. с sigmoid градиент затухает)
            x, y = next(iter(train_data))
            x = x / 127.5 - 1
            noise = np.random.normal(0, 1, (batch_size, self.LATENT_DIM))

            yield x, y, noise

    def sample_images(self, epoch):
        row, column = 2, self.NUM_CLASSES
        noise = np.random.normal(0, 1, (row * column, self.LATENT_DIM))
        label = np.array([
            np.arange(0, self.NUM_CLASSES) for _ in range(row)
        ]).reshape((-1, 1))
        sampled_labels = keras.utils.to_categorical(label, self.NUM_CLASSES)

        gen_imgs = self.generator.predict([noise, sampled_labels], verbose=False)
        gen_imgs = np.array(gen_imgs)
        gen_imgs -= np.min(gen_imgs)
        gen_imgs /= np.max(gen_imgs)

        # Делаем картинку
        fig, axs = plt.subplots(row, column, figsize=(12, 6))
        count = 0
        for i in range(row):
            for j in range(column):
                axs[i, j].imshow(gen_imgs[count, :, :, :])
                axs[i, j].set_title(label[count][0])
                axs[i, j].axis("off")
                count += 1
        fig.savefig("generated_flowers/%d.png" % epoch)
        plt.close()

    def train(self, batch_size=32, dataset="flowers_dataset"):
        # # Просто единицы и нули для Дискриминатора
        # valid = np.ones((batch_size, 1))
        # fake = np.zeros((batch_size, 1))
        #
        # get_batch = self.batch_gen(batch_size=batch_size, dataset=dataset)
        #
        # epoch_count = 1
        # all_l_dis = [0]
        # all_l_gen = [0]
        #
        # for learn_iter in range(int(10**10)):
        #     # Обучение дискриминатора
        #     for _ in range(1 + (self.HANDICAP if np.mean(all_l_dis)-0.15 > np.mean(all_l_gen) else 0)):
        #         images, labels, noise = next(get_batch)
        #         with tf.GradientTape() as dis_tape:
        #             dis_real_output = self.discriminator([images, labels], training=True)
        #             generated_images = self.generator([noise, labels], training=False)
        #             dis_fake_output = self.discriminator([generated_images, labels], training=True)
        #
        #             # Чем настоящие картинки нереальнее или сгенерированные реальные, тем ошибка больше
        #             l_dis = 0.5 * (tf.reduce_mean(-tf.math.log(dis_real_output + 1e-9)) +
        #                            tf.reduce_mean(-tf.math.log(1. - dis_fake_output + 1e-9)))
        #
        #         all_l_dis.append(l_dis)
        #
        #         # Получаем градиенты для дискриминатора
        #         grads_dis = dis_tape.gradient(l_dis, self.discriminator.trainable_variables)
        #         self.optimizer_dis.apply_gradients(zip(grads_dis, self.discriminator.trainable_variables))
        #
        #     # Обучение генератора
        #     for _ in range(1 + (self.HANDICAP if np.mean(all_l_gen)-0.15 > np.mean(all_l_dis) else 0)):
        #         images, labels, noise = next(get_batch)
        #         with tf.GradientTape() as gen_tape:
        #             generated_images = self.generator([noise, labels], training=True)
        #             dis_output = self.discriminator([generated_images, labels], training=False)
        #
        #             # Чем более реалистичная картина (для дискриминатора), тем меньше ошибка
        #             l_gen = -tf.reduce_mean(tf.math.log(dis_output + 1e-9))
        #
        #         all_l_gen.append(l_gen)
        #
        #         # Получаем градиенты для генератора
        #         grads_gen = gen_tape.gradient(l_gen, self.generator.trainable_variables)
        #         self.optimizer_gen.apply_gradients(zip(grads_gen, self.generator.trainable_variables))
        #
        #     # Сохраняем генерируемые образцы каждую эпоху
        #     if learn_iter % (2800 // batch_size) == 0:
        #         self.sample_images(epoch_count)
        #
        #         # Вывод прогресса и средних ошибок
        #         print(f"{epoch_count:02} \t"
        #               f"[Dis loss: {np.mean(all_l_dis):.3f}] \t"
        #               f"[Gen loss: {np.mean(all_l_gen):.3f}]")
        #
        #         epoch_count += 1
        #         all_l_dis = [0]
        #         all_l_gen = [0]

        train_data = keras.preprocessing.image_dataset_from_directory(
            dataset,
            image_size=self.IMG_SHAPE[:-1],
            label_mode="categorical",
            shuffle=True,
            batch_size=1,
        )

        valid = np.ones(shape=(2799, 1))
        fake = np.zeros(shape=(2799, 1))

        for i in range(1, 10**10):
            images = np.array([x[0]/255.*2 - 1 for x, y in train_data])
            labels = np.array([y[0] for x, y in train_data])
            noise = np.random.randint(0, 1, size=(2799, self.LATENT_DIM))

            generated_images = self.generator.predict([noise, labels])
            self.discriminator.fit([generated_images, labels], fake)
            self.discriminator.fit([images, labels], valid)
            self.ccgan.fit([noise, labels], valid)

            self.sample_images(i)


ccgan = CCGAN()
print("Generator:    ", f"{ccgan.generator.count_params():,}")
print("Discriminator:", f"{ccgan.discriminator.count_params():,}")
print("Sum:          ", f"{ccgan.generator.count_params() + ccgan.discriminator.count_params():,}")

ccgan.train(batch_size=32, dataset="flowers_dataset")
