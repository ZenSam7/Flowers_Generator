from keras.utils import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import os
import glob

data_augment = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.0,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="reflect"
)

path = "flowers"
save_to_dir = "new_flowers"
increase_coeff = 30  # Во сколько раз увеличиваем датасет


os.mkdir(save_to_dir)

count_prepearing_images = 1
count_all_images = sum([len(os.listdir("flowers/" + type_flower)) for type_flower in os.listdir("flowers")])
for directory_name in os.listdir(path):
    os.mkdir(save_to_dir + "\\" + directory_name)  # Сохраняем ярлыки (lable)

    for image_name in os.listdir(f"{path}\\{directory_name}"):
        # Загружаем изображение
        try:
            single_img = load_img(f"{path}\\{directory_name}\\{image_name}")
        except:
            continue
        image_array = img_to_array(single_img)
        image_array = image_array.reshape((1,) + image_array.shape)

        # И раздуваем его в increase_coeff раз
        i = 0
        for batch in data_augment.flow(
                image_array,
                save_to_dir=save_to_dir + "\\" + directory_name,
                save_prefix=directory_name,
                save_format="jpg"):
            i += 1
            if i == increase_coeff:
                break

        print(directory_name, count_prepearing_images,
              f"{int(count_prepearing_images / count_all_images * 100)}%")
        count_prepearing_images += 1
