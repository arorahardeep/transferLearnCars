#!/usr/bin/env python

"""
    This program defines a CNN in Keras to classify sports cars
    @author:    Hardeep Arora
    @date  :    17-Sep-2017
"""

# Libraries used
from car_utils import CarUtils
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Dropout
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.models import Model
from time import time
from matplotlib import pyplot as plt
from keras.applications import InceptionV3
from keras.applications import VGG19
from matplotlib.cbook import popall



class CarNet:
    """
        This class create the nn model
    """
    @staticmethod
    def build(input_shape, classes):
        """
            Builds the model topology

        :param input_shape: Shape of the data
        :param classes: Number of classes

        :return: model: The nn model definition
        """
        model = Sequential()
        model.add(Conv2D(20, kernel_size=9, padding="same", input_shape=input_shape,
                         kernel_initializer='glorot_normal'))

        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), dim_ordering="tf"))
        model.add(Dropout(0.3))

        model.add(Conv2D(40, kernel_size=5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), dim_ordering="tf"))
        model.add(Dropout(0.3))

        model.add(Conv2D(40, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(50, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(50, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), dim_ordering="tf"))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.3))

        model.add(Dense(512))
        model.add(Activation("relu"))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

    @staticmethod
    def build_inception(input_shape, classes):
        #Input = tf.contrib.keras.layers.Input(shape=input_shape)
        cnn = InceptionV3(weights='imagenet',
                        input_shape = input_shape,
                        include_top=False,
                        pooling='avg')
        for layer in cnn.layers:
            layer.trainable = False
        cnn.trainable = False
        x = cnn.output
        x = Dropout(0.6)(x)
        x = Dense(1024, activation='relu', name='dense01')(x)
        #x = Dropout(0.2)(x)
        x = Dense(512, activation='relu', name='dense02')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', name='dense03')(x)
        x = Dense(64, activation='relu', name='dense04')(x)
        Output = Dense(classes, activation='softmax', name='output')(x)
        return Model(cnn.input, Output)


class CarsClassifierModel:
    """
        This is the Cars Classifier Model class. This class defines the NN model constants and helper methods
    """

    # Class variables
    _nb_epoch = 250
    _batch_size = 64
    _verbose = 1
    _optimizer = Adam(lr=0.00001,decay=1e-7)
    _validation_split = 0.05
    _img_rows, _img_cols = 370, 370
    _nb_classes = 5
    _input_shape = ( _img_rows, _img_cols, 3)

    def __init__(self):
        self._model = CarNet.build_inception(self._input_shape, self._nb_classes)

    def plots(self, history):
        """
            This method plots the model performance graphs
        :return: nothing
        """
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()


    def save(self, filename):
        """
            This method saves the model definition and weights as *.h5
        :return: nothing
        """
        # serialize model to JSON
        model_json = self._model.to_json()
        with open('models/' + filename + ".json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self._model.save_weights('models/' + filename + ".h5")
        print("Saved model to disk")

    def evaluate(self, test_x, test_y):
        """
            This method evaluates the model performance
        :param test_x: Test data
        :param test_y: Test labels
        :return: nothing
        """
        score = self._model.evaluate(test_x, test_y, verbose=self._verbose)
        print("Test score: ", score[0])
        print("Test accuracy: ", score[1])

    def run(self, train_x, train_y):
        """
            This method runs the model
        :param train_x: Train data
        :param train_y: Train labels
        :return: history: model details
        """
        self._model.compile(loss="categorical_crossentropy", optimizer=self._optimizer, metrics=['accuracy'])
        self._model.summary()
        ## Install tensorboard support
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        history = self._model.fit(train_x,
                             train_y,
                             batch_size=self._batch_size,
                             epochs=self._nb_epoch,
                             verbose=self._verbose,
                             validation_split=self._validation_split,
                             callbacks=[tensorboard])
        return history

    def load_data_preprocess(self):
        """
            This method loads and pre-processes the data
        :return:    train_x: Train data
                    train_y: Train labels
                    test_x : Test data
                    test_y : Test labels
        """

        print("Loading the dataset ...")
        # load the data
        c_util = CarUtils()
        train_x, train_y, test_x, test_y, classes = c_util.load_data()

        # set the image ordering
        K.set_image_dim_ordering("th")

        print("Pre-processing the dataset ...")
        # pre-process the data
        train_x = train_x.astype('float32')
        test_x  = test_x.astype('float32')

        train_x = train_x / 255
        test_x  = test_x  / 255

        print(train_x.shape[0], ' train samples')
        print(test_x.shape[0], ' test samples')

        train_y = np_utils.to_categorical(train_y, CarsClassifierModel._nb_classes)
        test_y  = np_utils.to_categorical(test_y,  CarsClassifierModel._nb_classes)

        return train_x, train_y, test_x, test_y


def main():
    """
        This is the main method for this program
    :return:
    """
    nn = CarsClassifierModel()
    train_x, train_y, test_x, test_y = nn.load_data_preprocess()
    history = nn.run(train_x,train_y)
    nn.evaluate(test_x, test_y)
    nn.save("keras_nn_5")
    #nn.plots(history)
    #print(train_x.shape)
    #plt.imshow(train_x[52])
    #plt.title("Car")
    #plt.show()
    #print(train_y[52])

if __name__ == '__main__':
    main()