from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

initialX = 70
initialY = 50
deltaX = 170
deltaY = 150

#axis = 0
#newXstart = previousX + 120
#newDeltaX = previousXDelta + 120


def main():
    print("Hello World")
    im = Image.open("non_contrarie.bmp")
    #im.show()
    print(im.size)
    box = (70, 50, 170, 150)
    region = im.crop(box)
    region = region.convert('L')
    print(region.size)
    region.show()

    nextRightBox = (190, 50, 290, 150)
    nextRightRegion = im.crop(nextRightBox)
    nextRightRegion.show()

    secondNextBox = (310, 50, 410, 150)
    secondNextRegion = im.crop(secondNextBox)
    secondNextRegion.show()

    #plt.figure()
    #plt.imshow(region)


if __name__ == "__main__":
    main()


