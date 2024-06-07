from tensorflow import keras
import tensorflow as tf

from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter

from PIL import Image
import os

def make_gif():
    """Создаём гифку"""
    print("Делаем гифку...", end="\t")

    # Считываем все файлы изображений и сортируем их по имени
    images = sorted(
        [os.path.join("generated_flowers", img) for img in os.listdir("generated_flowers")],
        key=lambda path: int(path.split("/")[-1].split(".")[0])
    )

    # Загружаем первое изображение для инициализации графика
    first_image = Image.open(images[0])

    # Инициализация фигуры без осей
    fig, ax = plt.subplots()
    ax.axis("off")

    # Настройка параметров подграфика
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    # Устанавливаем размер фигуры точно соответствующий размеру изображения
    fig.set_size_inches(first_image.width / 100.0, first_image.height / 100.0)
    im = ax.imshow(first_image)

    # Функция для обновления изображения на каждом кадре
    def update(frame):
        image = Image.open(images[frame])
        im.set_data(image)
        return [im]

    # Создание анимации
    ani = FuncAnimation(fig, update, frames=len(images), blit=True)

    # Сохранение анимации в GIF файл
    ani.save("animation.gif", writer=PillowWriter(fps=5), dpi=200)
    print("Done")


def delete_images():
    # Удаляем все прошлые изображения
    for i in os.listdir("./generated_flowers"):
        os.remove(f"./generated_flowers/{i}")


class CCGAN(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.NUM_CLASSES = 5  # Не менять!

        # Входные форматы
        self.IMG_SHAPE = (120, 120, 3)
        self.LATENT_DIM = 8
        self.HANDICAP = 5  # Фора чтобы одна иишка не отставала от другой (только при self.LEARNING_TYPE = False)

        # Константы
        self.FILTERS_DIS = 32  # Нижняя граница
        self.FILTERS_GEN = 16
        self.DROPOUT = 0.2
        self.HIDDEN_IMG_SHAPE = (60, 60, 1)
        self.LEARNING_TYPE = True  # True == используя стандартный keras, False == кастомный метод обучения

        self.DISCRIMINATOR_LAYERS = 4
        self.DIS_CONS_LAYERS = 3

        self.GEN_DENSE_LAYERS = 8
        self.GEN_CONV_LAYERS = 2

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
        # z == latten_space (зображение между нейронками, которое мы выводим)
        self.generated_z = self.generator([self.latent_space_inp, self.label_inp])
        self.dis_gen_z = self.discriminator([self.generated_z, self.label_inp])

        self.ccgan = Model([self.latent_space_inp, self.label_inp], self.dis_gen_z, name="CCGAN")

        # self.LEARNING_TYPE == True
        self.discriminator.trainable = True
        self.discriminator.compile(Adam(5e-4), loss="binary_crossentropy")

        self.discriminator.trainable = False
        self.ccgan.compile(Adam(1e-3), loss="binary_crossentropy")

        # self.LEARNING_TYPE == False
        self.optimizer_gen = Adam(5e-4)
        self.optimizer_dis = Adam(5e-4)

    def build_discriminator(self) -> Model:
        # Объединяем картинку с лейблами
        # x = Embedding(self.NUM_CLASSES, self.IMG_SHAPE[0]**2)(self.label_inp)
        # x = Reshape([*self.IMG_SHAPE[:2], self.NUM_CLASSES])(x)
        # x = concatenate([BatchNormalization()(self.image_inp), x])
        x = BatchNormalization()(self.image_inp)

        for i in range(self.DISCRIMINATOR_LAYERS):
            x = BatchNormalization()(Dropout(self.DROPOUT)(x))
            for _ in range(self.DIS_CONS_LAYERS):
                x = Conv2D(self.FILTERS_DIS * 2**i, (3, 3), activation=LeakyReLU())(x)

            x = AveragePooling2D()(x)

        x = Flatten()(x)

        # Постепенно сжимаем
        dense_units = x.shape[-1] // 2
        while x.shape[-1] >= 8:
            x = Dropout(self.DROPOUT)(x)
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

        for _ in range(self.GEN_DENSE_LAYERS):
            x = BatchNormalization()(Dropout(self.DROPOUT)(x))
            x = concatenate([x, self.label_inp])
            x = Dense(x.shape[-1]*1.5, activation=LeakyReLU())(x)

        x = concatenate([x, self.label_inp])
        x = Dense(np.prod(self.HIDDEN_IMG_SHAPE), activation=LeakyReLU())(x)
        x = Reshape(self.HIDDEN_IMG_SHAPE)(x)

        x = UpSampling2D()(x)
        for _ in range(self.GEN_CONV_LAYERS):
            x = BatchNormalization()(Dropout(self.DROPOUT)(x))
            x = Conv2DTranspose(self.FILTERS_GEN, (3, 3), activation=LeakyReLU())(x)
            x = Conv2D(self.FILTERS_GEN, (3, 3), activation=LeakyReLU())(x)

        generated_img = Conv2D(3, (1, 1), activation="tanh")(x)

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
            x = x / 255. *2 - 1
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
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        count = 0
        for i in range(row):
            for j in range(column):
                axs[i, j].imshow(gen_imgs[count, :, :, :])
                axs[i, j].set_title(label[count][0])
                axs[i, j].axis("off")
                count += 1

        fig.savefig("generated_flowers/%d.png" % epoch, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

    def train(self, batch_size=32, dataset="flowers_dataset"):
        if self.LEARNING_TYPE:
            valid = np.ones(shape=(batch_size, 1))
            fake = np.zeros(shape=(batch_size, 1))

            get_batch = self.batch_gen(batch_size, dataset)

            for learn_iter in range(10 ** 10):
                # Загружаем случайные картинки и лейблы
                images, labels, noise = next(iter(get_batch))

                generated_images = self.generator.predict([noise, labels])
                self.discriminator.fit([generated_images, labels], fake, verbose=1)
                self.discriminator.fit([images, labels], valid, verbose=1)

                # Обучаем генератор
                self.discriminator.trainable = False
                self.ccgan.fit([noise, labels], valid, verbose=1)
                self.discriminator.trainable = True

                # Сохраняем генерируемые образцы каждую эпоху
                if learn_iter % 5 == 0:
                    self.sample_images(learn_iter // 5)

        else:
            # Просто единицы и нули для Дискриминатора
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            get_batch = self.batch_gen(batch_size=batch_size, dataset=dataset)

            epoch_count = 1
            all_l_dis = [0]
            all_l_gen = [0]

            for learn_iter in range(10**10):
                # Обучение дискриминатора
                for _ in range(1 + (self.HANDICAP if np.mean(all_l_dis)-0.15 > np.mean(all_l_gen) else 0)):
                    images, labels, noise = next(get_batch)
                    with tf.GradientTape() as dis_tape:
                        dis_real_output = self.discriminator([images, labels], training=True)
                        generated_images = self.generator([noise, labels], training=False)
                        dis_fake_output = self.discriminator([generated_images, labels], training=True)

                        # Чем настоящие картинки нереальнее или сгенерированные реальные, тем ошибка больше
                        l_dis = 0.5 * (tf.reduce_mean(-tf.math.log(dis_real_output + 1e-8)) +
                                       tf.reduce_mean(-tf.math.log(1. - dis_fake_output + 1e-8)))

                    all_l_dis.append(l_dis)

                    # Получаем градиенты для дискриминатора
                    grads_dis = dis_tape.gradient(l_dis, self.discriminator.trainable_variables)
                    self.optimizer_dis.apply_gradients(zip(grads_dis, self.discriminator.trainable_variables))

                # Обучение генератора
                for _ in range(1 + (self.HANDICAP if np.mean(all_l_gen)-0.15 > np.mean(all_l_dis) else 0)):
                    images, labels, noise = next(get_batch)
                    with tf.GradientTape() as gen_tape:
                        generated_images = self.generator([noise, labels], training=True)
                        dis_output = self.discriminator([generated_images, labels], training=False)

                        # Чем более реалистичная картина (для дискриминатора), тем меньше ошибка
                        l_gen = -tf.reduce_mean(tf.math.log(dis_output + 1e-8))

                    all_l_gen.append(l_gen)

                    # Получаем градиенты для генератора
                    grads_gen = gen_tape.gradient(l_gen, self.generator.trainable_variables)
                    self.optimizer_gen.apply_gradients(zip(grads_gen, self.generator.trainable_variables))

                # Сохраняем генерируемые образцы каждую эпоху
                if learn_iter % (2 * batch_size) == 0:
                    self.sample_images(epoch_count)

                    # Вывод прогресса и средних ошибок
                    print(f"{epoch_count:02} \t"
                          f"[Dis loss: {np.mean(all_l_dis):.3f}] \t"
                          f"[Gen loss: {np.mean(all_l_gen):.3f}]")

                    epoch_count += 1
                    all_l_dis = [0]
                    all_l_gen = [0]


ccgan = CCGAN()
print("Generator:    ", f"{ccgan.generator.count_params():,}")
print("Discriminator:", f"{ccgan.discriminator.count_params():,}")
print("Sum:          ", f"{ccgan.generator.count_params() + ccgan.discriminator.count_params():,}")


make_gif()
delete_images()

ccgan.train(batch_size=512, dataset="flowers_dataset")
