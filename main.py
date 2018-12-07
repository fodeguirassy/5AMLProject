# coding=utf-8
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

initialX = 70
initialY = 50
deltaX = 170
deltaY = 150


#axis = 0
#newXstart = previousX + 120
#newDeltaX = previousXDelta + 120


def main():
    print("Debut Traitement.")
    im = Image.open("non_contrarie.bmp")
    im.show()
    imgWidth, imgHeight = im.size
    print(imgWidth)
    print(imgHeight)

    #Variable Initialisation
    xbegin = 70
    ybegin = 170
    widthBox = 50
    heightBox = 150
    rowMax = 3
    colMax = 3
    dataArray = [[0 for x in range(rowMax)] for y in range(colMax)]



    # Parcours de la feuille
    indexRow = 0
    indexCol = 0
    while indexCol < colMax:
        while indexRow < rowMax:
            box = (xbegin, widthBox, ybegin, heightBox)
            region = im.crop(box)
            region = region.convert('L')
            dataArray[indexRow][indexCol] = region
            #region.show()
            xbegin += 120
            ybegin += 120
            indexRow += 1
        indexCol += 1
        indexRow = 0
        widthBox += 120
        heightBox += 120


    dataArray[0][2].show()

    #plt.figure()
    #plt.imshow(region)


if __name__ == "__main__":
    main()


