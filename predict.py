# coding: utf-8

"""
this code is to predict input image label
@ shumpei hatanaka
"""

import sys
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import tensorflow
from PIL import Image
from generate_data import get_classes


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(50, 50, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']
                  )
    model.summary()

    model = load_model("./model/model_aug.h5")

    return model

def main():
    classes = get_classes()
    image = Image.open(sys.argv[1])
    image = image.convert("RGB")
    image = image.resize((50, 50))
    data = np.asarray(image) / 255
    x = []
    x.append(data)
    x = np.array(x)
    model = build_model()

    result = model.predict([x])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    print(f"{classes[predicted]} ({percentage} %)")


if __name__ == '__main__':
    main()
