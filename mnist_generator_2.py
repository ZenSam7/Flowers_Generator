
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

# Загрузка данных MNIST
(train_images, _), (test_images, _) = datasets.mnist.load_data()

# Нормализация изображений к диапазону [0, 1]
train_images = train_images.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

# Преобразование изображений в вектора размерности 784 (28x28)
train_images = train_images.reshape(train_images.shape[0], 784)
test_images = test_images.reshape(test_images.shape[0], 784)

# Создание вариационного автоэнкодера
latent_dim = 2

encoder_inputs = tf.keras.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(encoder_inputs)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(256, activation='relu')(latent_inputs)
outputs = layers.Dense(784, activation='sigmoid')(x)

decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

outputs = decoder(encoder(encoder_inputs)[2])
vae = tf.keras.Model(encoder_inputs, outputs, name='vae')

# Функция потерь вариационного автоэнкодера
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Компиляция модели
vae.compile(optimizer='adam')

# Обучение модели
vae.fit(train_images, train_images, epochs=15, batch_size=32)

# Генерация изображений с помощью вариационного автоэнкодера
encoded_imgs = encoder.predict(test_images)[2]
decoded_imgs = decoder.predict(encoded_imgs)

# Отображение оригинальных и восстановленных изображений
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Оригинальные изображения
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Восстановленные изображения
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
