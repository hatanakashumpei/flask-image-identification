# coding: utf-8

"""
this code is generate numpy dataset
@ shumpei hatanaka
"""

import glob
import numpy as np
from PIL import Image


def get_classes():
    """get class names

    Returns:
        list: class names
    """
    data_list_paths = list(glob.glob("./data/*"))
    class_names = []
    for data_list_path in data_list_paths:
        name = data_list_path.replace("./data/", "")
        class_names.append(name)

    return class_names


def generate_dataset(labels, num):
    """generate dataset with DA

    Args:
        labels (list): list of classes

    Returns:
        numpy: x_data(image), y_data(class label)
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    image_size = 50
    for index, label in enumerate(labels):
        photos_dir = f"./data/{label}/*.jpg"
        files = glob.glob(photos_dir)
        for i, file in enumerate(files):
            if i >= 130:
                break
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image, dtype=object)
            # split test/train data
            if i < num:
                x_test.append(data)
                y_test.append(index)
            else:
                x_train.append(data)
                y_train.append(index)

                # Data Augmentation
                for angle in range(-20, 20, 5):
                    # rotate
                    img_r = image.rotate(angle)
                    data = np.asarray(img_r)
                    x_train.append(data)
                    y_train.append(index)

                    # tranpose
                    img_tra = image.transpose(Image.FLIP_LEFT_RIGHT)
                    data = np.asarray(img_tra)
                    x_train.append(data)
                    y_train.append(index)

    x_train = np.array(x_train, dtype=object)
    y_train = np.array(y_train, dtype=object)
    x_test = np.array(x_test, dtype=object)
    y_test = np.array(y_test, dtype=object)

    return x_train, x_test, y_train, y_test


def cretate_dataset(x_train, x_test, y_train, y_test):
    """create dataset
    """
    dataset = (x_train, x_test, y_train, y_test)
    np.save("./dataset_aug.npy", dataset)


def main():
    """Run generate_data.py
    """
    classes = get_classes()
    # num_classes = len(classes)
    num_testdata = 50
    x_train, x_test, y_train, y_test = generate_dataset(classes, num_testdata)
    cretate_dataset(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    print(get_classes())
    main()
