from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Lambda
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import tensorflow as tf

# Загрузка данных MNIST
(x_train, _), (x_test, _) = mnist.load_data()

# Нормализация и изменение формы данных
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Создание модели автоэнкодера
input_img = Input(shape=(28, 28, 1))

units = 8
filters = 64
batch_size = 64

x = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(units, activation="sigmoid")(x)

z_mean = Dense(units)(x)
z_log_var = Dense(units)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], units), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

l = Lambda(sampling, output_shape=(units,))([z_mean, z_log_var])

encoder = Model(input_img, l, name="encoder")

latent_inputs = Input(shape=(units,))
x = Reshape((2, 2, units//4))(latent_inputs)
x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(filters, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
decoder = Model(latent_inputs, decoded_output, name="decoder")

autoencoder_output = decoder(encoder(input_img))
autoencoder = Model(input_img, autoencoder_output)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# Обучение модели
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))

# Проверка автоэнкодера
decoded_imgs = autoencoder.predict(x_test)

# Отображение результатов
import matplotlib.pyplot as plt
from random import randint

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    rand_num = randint(0, len(decoded_imgs)-1)
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[rand_num].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[rand_num].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Генерация новых изображений
decoded_imgs = decoder_generator.predict(np.random.random((999, units)))

plt.figure(figsize=(20, 4))
for i in range(n):
    rand_num = randint(0, len(decoded_imgs))
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(decoded_imgs[rand_num].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    rand_num = randint(0, len(decoded_imgs))
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[rand_num].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
