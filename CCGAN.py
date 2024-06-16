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

flowers_path = "./generated_flowers"


def make_gif():
    """Создаём гифку"""
    print("Делаем гифку...", end="\t")

    # Считываем все файлы изображений и сортируем их по имени
    images = sorted(
        [os.path.join(flowers_path, img) for img in os.listdir(flowers_path)],
        key=lambda path: int(path.split("\\")[-1].split(".")[0])
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
    for i in os.listdir(flowers_path):
        os.remove(f"{flowers_path}/{i}")


class CCGAN(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.NUM_CLASSES = 5  # Не менять!

        # Входные форматы
        self.IMG_SHAPE = (128, 128, 3)
        self.LATENT_DIM = 8
        self.HANDICAP = 5  # Фора чтобы одна иишка не отставала от другой (только при self.LEARNING_TYPE == 0)

        # Константы
        self.FILTERS_DIS = 3   # Нижняя граница
        self.FILTERS_GEN = 3   # Нижняя граница
        self.HIDDEN_IMG_SHAPE = (4, 4, 1)
        # "keras" == используя стандартный keras, "колхоз" == кастомный метод обучения
        self.LEARNING_TYPE = "колхоз"

        self.DIS_LAYERS = 6
        self.GEN_LAYERS = 5

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

        if self.LEARNING_TYPE == "keras":
            self.discriminator.trainable = True
            self.discriminator.compile(Adam(1e-4), loss="binary_crossentropy")

            self.discriminator.trainable = False
            self.ccgan.compile(Adam(1e-4), loss="binary_crossentropy")

        else:
            self.optimizer_gen = Adam(1e-3)
            self.optimizer_dis = Adam(1e-3)

    def build_discriminator(self) -> Model:
        # Объединяем картинку с лейблами
        units_repeat = RepeatVector(int(np.prod(self.IMG_SHAPE)))(self.label_inp)
        units_repeat = Reshape([*self.IMG_SHAPE[:-1], self.IMG_SHAPE[-1]*self.NUM_CLASSES])(units_repeat)

        norm_image = BatchNormalization()(self.image_inp)
        x = concatenate([units_repeat, norm_image])

        for i in range(self.DIS_LAYERS):
            x = Conv2D(self.FILTERS_DIS * 2**i, (3, 3), padding="same", activation=LeakyReLU())(x)
            x = Conv2D(self.FILTERS_DIS * 2**i, (3, 3), padding="same", activation=LeakyReLU())(x)
            x = Conv2D(self.FILTERS_DIS * 2**i, (3, 3), padding="same", activation=LeakyReLU())(x)
            x = AveragePooling2D()(x)
            x = BatchNormalization()(x)

        # Сокращаем всё до вектора
        x = GlobalAveragePooling2D()(x)

        # Добавляем метки класса
        x = Flatten()(x)
        x = concatenate([self.label_inp, x])
        x = Dense(1, activation="sigmoid")(x)

        self.discriminator = Model([self.image_inp, self.label_inp], x, name="discriminator")

    def build_generator(self) -> Model:
        # Разжимаем вектор шума в маленькую картинку
        x = concatenate([self.latent_space_inp, self.label_inp])
        x = Dense(np.prod(self.HIDDEN_IMG_SHAPE))(x)
        x = Reshape(self.HIDDEN_IMG_SHAPE)(x)

        for i in range(self.GEN_LAYERS - 1, -1, -1):
            x = Conv2DTranspose(self.FILTERS_GEN * 2**i, (9, 9), padding="same", activation="tanh")(x)
            x = UpSampling2D()(x)
            x = BatchNormalization()(x)

        x = Conv2D(3, (1, 1), activation="tanh", padding="same")(x)
        generated_img = BatchNormalization()(x)

        self.generator = Model([self.latent_space_inp, self.label_inp], generated_img, name="generator")

    def batch_gen(self, dataset):
        """Чтобы использовать "big_flowers_dataset" (расширенный датасет) надо запустить increasing_data.py"""
        train_data = keras.preprocessing.image_dataset_from_directory(
            dataset,
            image_size=self.IMG_SHAPE[:-1],
            label_mode="categorical",
            shuffle=True,
            batch_size=self.batch_size,
        )

        while True:
            # Добавляем лейблы (т.к. у нас CCGAN) и нормализуем в [-1; 1], т.к. юзаем tanh
            # (т.к. с sigmoid градиент затухает)
            x, y = next(iter(train_data))
            x = x / 255. * 2 - 1
            yield x, y

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

    @tf.function
    def train_step(self, images, labels):
        noise = tf.random.normal((self.batch_size, self.LATENT_DIM))

        with tf.GradientTape() as dis_tape, tf.GradientTape() as gen_tape:
            generated_images = self.generator([noise, labels], training=True)
            dis_real_output = self.discriminator([images, labels], training=True)
            dis_fake_output = self.discriminator([generated_images, labels], training=True)

            # Чем настоящие картинки нереальнее или сгенерированные реальнее, тем ошибка больше
            l_dis = 0.5 * (tf.reduce_mean(-tf.math.log(dis_real_output + 1e-8)) +
                           tf.reduce_mean(-tf.math.log(1. - dis_fake_output + 1e-8)))

            # Чем более реалистичная картина (для дискриминатора), тем меньше ошибка
            l_gen = tf.reduce_mean(-tf.math.log(dis_fake_output + 1e-8))

        # Получаем и применяем градиенты
        grads_dis = dis_tape.gradient(l_dis, self.discriminator.trainable_variables)
        self.optimizer_dis.apply_gradients(zip(grads_dis, self.discriminator.trainable_variables))

        grads_gen = gen_tape.gradient(l_gen, self.generator.trainable_variables)
        self.optimizer_gen.apply_gradients(zip(grads_gen, self.generator.trainable_variables))

        return l_gen, l_dis

    def train(self, batch_size=32, dataset="flowers_dataset"):
        self.batch_size = batch_size
        get_batch = self.batch_gen(dataset)

        if self.LEARNING_TYPE == "keras":
            valid = np.ones(shape=(batch_size, 1))
            fake = np.zeros(shape=(batch_size, 1))

            for learn_iter in range(10 ** 10):
                # Сохраняем генерируемые образцы каждую эпоху
                if learn_iter % 10 == 0:
                    self.sample_images(learn_iter // 10)

                # Загружаем случайные картинки и лейблы
                noise = np.random.normal(0, 1, (self.batch_size, self.LATENT_DIM))
                images, labels = next(iter(get_batch))

                generated_images = self.generator.predict([noise, labels])
                self.discriminator.fit([generated_images, labels], fake, verbose=1)
                self.discriminator.fit([images, labels], valid, verbose=1)

                # Обучаем генератор
                self.ccgan.fit([noise, labels], valid, verbose=1)

        """LEARNING_TYPE == колхоз"""
        # Просто единицы и нули для Дискриминатора
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        get_batch = self.batch_gen(dataset=dataset)

        epoch_count = 1
        all_l_dis = [0]
        all_l_gen = [0]

        for learn_iter in range(10 ** 10):
            # Обучение дискриминатора
            for _ in range(1 + (self.HANDICAP if np.mean(all_l_dis) - 0.15 > np.mean(all_l_gen) else 0)):
                images, labels = next(get_batch)
                noise = np.random.normal(0, 1, [batch_size, self.LATENT_DIM])
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
            for _ in range(1 + (self.HANDICAP if np.mean(all_l_gen) - 0.15 > np.mean(all_l_dis) else 0)):
                images, labels = next(get_batch)
                noise = np.random.normal(0, 1, [batch_size, self.LATENT_DIM])
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
            if learn_iter % 10 == 0:
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

# try:
#     make_gif()
# except IndexError:
#     print("Нет изображений")
delete_images()

ccgan.train(batch_size=100, dataset="big_flowers_dataset")
