# coding: utf-8

"""
this code is generate numpy dataset
@ shumpei hatanaka
"""

import os
import numpy as np
import glob
from PIL import Image
from sklearn import model_selection


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


def dataset(labels):
    """generate dataset

    Args:
        labels (list): list of classes

    Returns:
        numpy: x_data(image), y_data(class label)
    """
    x_data = []
    y_data = []
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
            data = np.asarray(image)
            x_data.append(data)
            y_data.append(index)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def main():
    classes = get_classes()
    num_classes = len(classes)
    x, y = dataset(classes)


if __name__ == '__main__':
    print(get_classes())
    main()
