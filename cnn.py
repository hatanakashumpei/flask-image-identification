# coding: utf-8

"""
this code is to define cnn model
@ shumpei hatanaka
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import tensorflow
from generate_data import get_classes


def model_train(x_train, y_train):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
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

    model.fit(x_train, y_train, batch_size=16, epochs=100)

    model.save("./model/model.h5")

    return model


def model_eval(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test Loss : {scores[0]}')
    print(f'Test Accuracy : {scores[1]}')



def main():
    num_classes = len(get_classes())
    print(num_classes)
    x_train, x_test, y_train, y_test = np.load("./dataset.npy", allow_pickle=True)
    x_train = x_train.astype("float") / 256
    x_test = x_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(x_train, y_train)
    model_eval(model, x_test, y_test)


if __name__ == '__main__':
    main()
