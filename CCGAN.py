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


class CCGAN():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.NUM_CLASSES = 5  # Не менять!

        # Входные форматы
        self.IMG_SHAPE = (160, 160, 3)
        self.LATENT_DIM = 16

        # Константы
        self.FILTERS_DIS = 16  # Нижняя граница

        self.DIS_LAYERS = 6
        self.GEN_LAYERS = 6

        self.optimizer_gen = Adam(4e-4)
        self.optimizer_dis = Adam(1e-4)

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

    def build_discriminator(self) -> Model:
        # Объединяем картинку с лейблами
        x = Embedding(self.NUM_CLASSES, self.IMG_SHAPE[0] ** 2)(self.label_inp)
        x = Reshape([*self.IMG_SHAPE[:2], self.NUM_CLASSES])(x)
        x = concatenate([self.image_inp, x])

        for i in range(self.DIS_LAYERS):
            x = Conv2D(self.FILTERS_DIS * 2 ** i, (3, 3), padding="same", activation="relu")(x)
            x = Conv2D(self.FILTERS_DIS * 2 ** i, (3, 3), padding="same", activation="relu")(x)
            x = AveragePooling2D()(x)

        x = Conv2D(x.shape[-1] * 2, x.shape[1:3], activation="relu")(x)

        x = Flatten()(x)
        while x.shape[-1] > 8:
            x = concatenate([self.label_inp, x])
            x = Dense(x.shape[-1] // 2, activation="relu")(x)

        x = Flatten()(x)
        # Добавляем метки класса
        x = concatenate([self.label_inp, x])
        predict = Dense(1, activation="sigmoid")(x)
        self.discriminator = Model([self.image_inp, self.label_inp], predict, name="discriminator")

    def build_generator(self) -> Model:
        # Разжимаем вектор шума в маленькую картинку
        x = self.latent_space_inp
        for _ in range(self.GEN_LAYERS):
            x = concatenate([x, self.label_inp])
            x = Dense(x.shape[-1] - self.NUM_CLASSES, activation="tanh")(x)

        # TODO: Никаких нахуй свёрток и развёрток!
        x = concatenate([x, self.label_inp])
        x = Dense(np.prod(self.IMG_SHAPE), activation="tanh")(x)
        x = Reshape(self.IMG_SHAPE)(x)

        generated_img = Conv2D(3, (3, 3), activation="tanh", padding="same")(x)
        self.generator = Model([self.latent_space_inp, self.label_inp], generated_img, name="generator")

    def train_data(self, dataset):
        """Чтобы использовать "big_flowers_dataset" (расширенный датасет) надо запустить increasing_data.py"""
        train_data = keras.preprocessing.image_dataset_from_directory(
            dataset,
            image_size=self.IMG_SHAPE[:-1],
            label_mode="categorical",
            shuffle=True,
            batch_size=self.batch_size,
        )

        return train_data

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
    def train_step(self, imgs, labels):
        noise = tf.random.normal((self.batch_size, self.LATENT_DIM))
        images = imgs / 127.5 - 1

        with tf.GradientTape() as dis_tape:
            generated_images = self.generator([noise, labels], training=False)
            dis_real_output = self.discriminator([images, labels], training=True)
            dis_fake_output = self.discriminator([generated_images, labels], training=True)

            # Чем настоящие картинки нереальнее или сгенерированные реальнее, тем ошибка больше
            l_dis = (tf.reduce_mean(-tf.math.log(dis_real_output + 1e-8)) +
                     tf.reduce_mean(-tf.math.log(1. - dis_fake_output + 1e-8)))

        with tf.GradientTape() as gen_tape:
            generated_images = self.generator([noise, labels], training=True)
            dis_fake_output = self.discriminator([generated_images, labels], training=False)

            # Чем более реалистичная картина (для дискриминатора), тем меньше ошибка
            # И учим Генератор делать просто более реалистичные изображения
            l_gen = tf.reduce_mean(-tf.math.log(dis_fake_output + 1e-8)) + \
                    0.1*tf.losses.mae(images, generated_images)

        # Получаем и применяем градиенты
        grads_dis = dis_tape.gradient(l_dis, self.discriminator.trainable_variables)
        self.optimizer_dis.apply_gradients(zip(grads_dis, self.discriminator.trainable_variables))

        grads_gen = gen_tape.gradient(l_gen, self.generator.trainable_variables)
        self.optimizer_gen.apply_gradients(zip(grads_gen, self.generator.trainable_variables))

        return l_gen, l_dis

    def train(self, batch_size=1, dataset="flowers_dataset"):
        self.batch_size = batch_size
        train_data = self.train_data(dataset)
        get_batch = iter(train_data)
        all_l_dis, all_l_gen = [], []  # Все ошибки

        for epoch in range(1, 10 ** 10):
            for _ in range(1000 // batch_size):
                try:
                    images, labels = next(get_batch)
                    l_gen, l_dis = self.train_step(images, labels)
                    all_l_gen.append(l_gen)
                    all_l_dis.append(l_dis)
                except Exception as e:
                    # Так надо, чтобы использовать любой batch_size
                    get_batch = iter(train_data)

            # Сохраняем генерируемые образцы каждую эпоху
            self.sample_images(epoch)

            # Вывод прогресса и средних ошибок
            print(f"{epoch:02} \t"
                  f"[Dis loss: {np.mean(all_l_dis):.3f}] \t"
                  f"[Gen loss: {np.mean(all_l_gen):.3f}]")

            # Останавливаем обучение, если что-то идёт не так
            if np.isnan(np.mean(all_l_dis)) or np.isnan(np.mean(all_l_gen)):
                exit()

            all_l_dis.clear()
            all_l_gen.clear()


ccgan = CCGAN()
print("Generator:    ", f"{ccgan.generator.count_params():,}")
print("Discriminator:", f"{ccgan.discriminator.count_params():,}")
print("Sum:          ", f"{ccgan.generator.count_params() + ccgan.discriminator.count_params():,}")

# try:
#     make_gif()
# except IndexError:
#     print("Нет изображений")
delete_images()

ccgan.train(batch_size=1, dataset="flowers_dataset")
