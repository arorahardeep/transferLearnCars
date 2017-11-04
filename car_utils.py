#!/usr/bin/env python3

from skimage import io as skio
from skimage import transform as transform
import matplotlib.pyplot as plt
import glob
import os
import numpy as np


class CarUtils:

    def load_image(self, path, size=370):
        img = skio.imread(path)
        resized_img = transform.resize(img, (size, size))
        return resized_img

    def crop_middle_square_area(np_image):
        h, w, _ = np_image.shape
        h = int(h/2)
        w = int(w/2)
        if h>w:
            return np_image[ h-w:h+w, : ]
        return np_image[ :, w-h:w+h ]

    def _read_folder(self, foldername,label):
        car_dataset = [self.load_image(path) for path in glob.glob(os.path.join(foldername, '*.jpg'))]
        car_label = label * len(car_dataset)
        return car_dataset, car_label

    def _load_dataset(self, setname):
        carlab = []
        carset = []

        class_name = ['ferrari', 'bugati', 'lambo', 'pagani', 'bmw']
        lbl = ['0','1','2','3','4']

        car_dir = ['dataset/' + setname + '/' + cl for cl in class_name]
        carsets = [self._read_folder(f,l) for f,l in zip(car_dir, lbl) ]

        for car,lab in carsets:
            carlab.extend(list(lab))
            carset.extend(car)

        classes= np.array(class_name[:])
        train_x = np.array(carset[:])
        train_y = np.array(carlab[:])

        return train_x, train_y, classes


    def load_data(self):
        train_x, train_y, classes = self._load_dataset('train')
        test_x, test_y, classes = self._load_dataset('test')
        return train_x, train_y, test_x, test_y, classes

def main():
    c_util = CarUtils()
    train_x, train_y, test_x, test_y, classes = c_util.load_data()

    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)

if __name__ == '__main__':
    main()