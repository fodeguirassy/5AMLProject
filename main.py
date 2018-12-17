from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

initialX = 70
initialY = 50
deltaX = 170
deltaY = 150

#axis = 0
#newXstart = previousX + 120
#newDeltaX = previousXDelta + 120


def main():
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
        current_models = []
        for j in range(col_max):
            current_box = (x_start, y_start, x_start + x_delta, y_start + y_delta)
            current_region = im.crop(current_box)
            current_region = current_region.convert('L')
            #current_region = current_region.resize((42, 38), Image.ANTIALIAS)
            #print("current region size {}".format(current_region.size))
            #current_region.show()

            pix = np.array(current_region)
            pix = pix.reshape((current_region.size[0] * current_region.size[1],))
            current_models.append(pix)
            #models_array.append(pix)

            #print("Current Region to numpy Array {}".format(pix))

            x_start = x_start + 120

        x_start = 70
        y_start = y_start + 120

        models_array.append(current_models)

    print("Models Array {}".format(np.array(models_array).shape))
    labels_array = np.array(labels_array)
    print("Models Array {}".format(labels_array.shape))

    #zipped_array = np.array(list(zip(models_array, labels_array)))
    #print("Zipped Array {}".format(zipped_array[10]))

    classifier = MLPClassifier().fit(models_array, labels_array)
    print(classifier.predict([models_array[10]]))


if __name__ == "__main__":
    main()


