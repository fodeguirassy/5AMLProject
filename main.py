from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier


# axis = 0
# newXstart = previousX + 120
# newDeltaX = previousXDelta + 120


def crop(labelsOnly):

    im = Image.open("non_contrarie.bmp")

    x_start = 70
    y_start = 50
    x_delta = 100
    y_delta = 100
    row_max = 19
    col_max = 10

    for i in range(18):
        current_box = (x_start, y_start, x_start + x_delta, y_start + y_delta)
        current_region = im.crop(current_box)
        current_region = current_region.convert('L')
        current_region.show()
        y_start = y_start + 120


def crop_all():

    im = Image.open("non_contrarie.bmp")

    x_start = 70
    y_start = 50
    x_delta = 100
    y_delta = 100
    row_max = 19
    col_max = 10

    models_array = []
    labels_array = ["a", "b", "c", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]

    for i in range(row_max):

        current_model = []

        for j in range(col_max):
            current_box = (x_start, y_start, x_start + x_delta, y_start + y_delta)
            current_region = im.crop(current_box)
            current_region = current_region.convert('L')

            pix = np.array(current_region)
            pix = pix.reshape((current_region.size[0] * current_region.size[1],))
            current_model.append(pix)

            x_start = x_start + 120
        x_start = 70
        y_start = y_start + 120
        models_array.append(current_model)

    models_array = np.array(models_array)
    models_array = models_array.reshape(models_array.shape[0],  models_array.shape[1] * models_array.shape[2])
    #models_array = models_array.transpose()

    classifier = MLPClassifier().fit(models_array, labels_array)
    print(classifier.predict([models_array[1]]))


def main():
    crop_all()


if __name__ == "__main__":
    main()
