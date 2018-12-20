from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import svm
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import re
from sklearn import metrics


def kppv(x, y, new_point, k):
    def dist(a, b):
        return np.linalg.norm(a-b)

    labeled_points = zip(x, y)
    by_distance = sorted(labeled_points, key=lambda point_label: dist(point_label[0], new_point))
    k_nearest_labels = [label for _, label in by_distance[:k]]
    print((Counter(k_nearest_labels).most_common()[0][0]))


def crop_all():

    im = Image.open("non_contrarie.bmp")

    x_start = 70
    y_start = 50
    x_delta = 100
    y_delta = 100
    row_max = 20
    col_max = 11

    labels_array = []
    models_array = []
    raw_labels_array = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]

    for i in range(row_max):

        current_model = []

        for j in range(col_max):
            current_box = (x_start, y_start, x_start + x_delta, y_start + y_delta)
            current_region = im.crop(current_box)
            current_region = current_region.convert('L')

            pix = np.array(current_region)
            pix = pix.reshape((current_region.size[0] * current_region.size[1],))

            current_model.append(pix)
            #models_array.append(pix)
            #labels_array.append(raw_labels_array[i])
            x_start = x_start + 120
        x_start = 70
        y_start = y_start + 120
        models_array.append(current_model)

    models_array = np.array(models_array)
    print("models Array shape {}" .format(models_array.shape))
    print("Labels Array shape {}" .format(np.array(raw_labels_array).shape))
    models_array = models_array.reshape(models_array.shape[0],  models_array.shape[1] * models_array.shape[2])

    zipped_array = np.array(list(zip(models_array, raw_labels_array)))
    print("Zipped Array {}" .format(zipped_array))

    #crop_single(6)
    #print("Test model shape {}".format(np.array(models_array[0]).shape))

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

#    for clf in classifiers:
#        clf.fit(x_train, y_train)
#        print("current score {}".format(clf.score(x_test, y_test)))

    #data = models_array[5]
    #data = data.reshape((data.shape[0], -1))
    #currImg = Image.fromarray(data, 'RGB')
    #currImg.show()

    #x_train, x_test, y_train, y_test = train_test_split(models_array, raw_labels_array, test_size=.4)

    crop_single_copy(0)

    clf = neighbors.KNeighborsClassifier(n_neighbors=1).fit(models_array, raw_labels_array)
    print("Score {}".format(metrics.accuracy_score(raw_labels_array, clf.predict(models_array))))
    #print("Test shape {}".format(models_array[5].shape))
    #clf = neighbors.KNeighborsClassifier(n_neighbors=2).fit(models_array, raw_labels_array)
    new = models_array[3]
    print("new value is {}".format(new.shape))
    #print(clf.predict([models_array[0, 3].reshape(1, -1)]))

    #print(clf.predict([crop_single(14)]))
    #clf = clf.fit(x_train, y_train)
    #print("score is {}".format(clf.score(x_test, y_test)))

    #classifier = MLPClassifier().fit(models_array, raw_labels_array)
    #print(classifier.predict([models_array[8]]))

    #print("Test shape {}".format(np.array(models_array[6]).shape))
    #crop_single(6)

    #kppv(models_array, raw_labels_array, models_array[9], 3)


def crop_single(index):

    im = Image.open("non_contrarie.bmp")

    x_start = 70
    y_start = 50
    x_delta = 100
    y_delta = 100
    row_max = 19
    col_max = 10

    result = []

    for i in range(19):

        if i == index:
            current_box = (x_start, y_start, x_start + x_delta, y_start + y_delta)
            current_region = im.crop(current_box)
            current_region = current_region.convert('L')
            current_region.show()

            pix = np.array(current_region)
            #pix = pix.reshape((test.shape[1] * test.shape[2]))
            pix = pix.reshape((current_region.size[0],) + current_region.size[1:])
            result.append(pix)
            break
        y_start = y_start + 120

    result = np.array(result)
    result = result.reshape(result.shape[1] * result.shape[2])
    print("Test shape {}".format(result.shape).replace(' ', ','))
    return result


def crop_single_copy(index):

    im = Image.open("non_contrarie.bmp")

    x_start = 70
    y_start = 50
    x_delta = 100
    y_delta = 100
    row_max = 19
    col_max = 10

    result = []

    for i in range(19):

        if i == index:
            current_box = (x_start, y_start, x_start + x_delta, y_start + y_delta)
            current_region = im.crop(current_box)
            current_region = current_region.convert('L')
            current_region.show()

            pix = np.array(current_region)
            #np.set_printoptions(threshold=np.nan)
            #pix = pix.reshape((test.shape[1] * test.shape[2]))
            pix = pix.reshape((current_region.size[0] * current_region.size[1],))
            #pix_string = "{}".format(pix)
            #print("Pix {}".format(re.sub("\s+", ",", pix_string.strip())))
            result.append(pix)
            break
        y_start = y_start + 120

    result = np.array(result)
    #result = result.reshape(result.shape[1] * result.shape[2])
    print("Test shape {}".format(result.shape))
    return result


def main():
    crop_all()
    #crop_single(index=6)


if __name__ == "__main__":
    main()
