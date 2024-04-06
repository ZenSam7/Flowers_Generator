from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from keras.datasets import mnist
import numpy as np

# Загрузка данных MNIST
(x_train, _), (x_test, _) = mnist.load_data()

# Нормализация и изменение формы данных
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Создание модели автоэнкодера
input_img = Input(shape=(28, 28, 1))

encoder = Sequential()
encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Flatten())

units = 16
encoder.add(Dense(units, activation="sigmoid"))

encoded_img = encoder(input_img)

decoder = Sequential()
decoder.add(Reshape((2, 2, units//4)))
decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(4, (3, 3), activation='relu'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

decoded_img = decoder(encoded_img)

autoencoder = Model(input_img, decoded_img)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

decoder_generator = Model(encoded_img, decoded_img)
decoder_generator.compile(optimizer='adam', loss='binary_crossentropy')

# Обучение модели
autoencoder.fit(x_train, x_train,
                epochs=20, 
                batch_size=128,
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
    rand_num = randint(0, len(decoded_imgs))
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
decoded_imgs = decoder_generator.predict(np.random.random((100, units)))
plt.figure(figsize=(20, 4))
for i in range(n):
    rand_num = randint(0, len(decoded_imgs))
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