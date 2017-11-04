#!/usr/bin/env python3

"""
    This program loads the Car Classifier model and uses it to predict new images
    @author:    Hardeep Arora
    @date  :    17-Sep-2017
"""

# Libraries used
from keras.models import model_from_json
from keras.optimizers import Adam
from car_utils import CarUtils
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


class CarClassifier:
    """
        This is the Cars Classifier class and contains helper methods
    """
    class_names = ['ferrari', 'bugati', 'lambo', 'pagani', 'bmw']

    def __init__(self, model_fname):
        self._load_model(model_fname)

    def _load_model(self, filename):
        """
            Loads the model
        :return: nothing
        """

        # load json and create model
        json_file = open('models/' + filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self._model = model_from_json(loaded_model_json)
        # load weights into new model
        self._model.load_weights('models/' + filename + ".h5")
        print("Loaded model from disk")
        self._model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        self._model.summary()


    def classify(self, path):
        """
            This method helps classify the cars images
            :return:
        """

        c_util = CarUtils()
        for filename in glob.glob(os.path.join(path, '*.jpg')):
            car_image = c_util.load_image(filename)
            car_image = np.array(car_image)
            car_image_prd = car_image[ np.newaxis,:, :, :]
            car_image_prd = car_image_prd/255
            #print(car_image_prd.shape)
            pred = self._model.predict( car_image_prd, batch_size=64, verbose=0)


            print(filename)
            print("This is " + self.class_names[np.argmax(pred[0])] + " and with %2.1f "%(pred[0][np.argmax(pred[0])]*100) + "% probability.")
            print(pred[0])

            plt.imshow(car_image)
            plt.title(filename)
            plt.show()


def main():
    """
    This methods instantiates the CarClassifier and run the image classification
    :return: Nothing
    """
    cars = CarClassifier("keras_nn_5")
    for name in CarClassifier.class_names:
        cars.classify('dataset/test/' + name)


if __name__ == '__main__':
    main()